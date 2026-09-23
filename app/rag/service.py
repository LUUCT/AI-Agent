"""文件业务编排：上传→MinIO→解析→按需向量化→旧接口/Agent 读取。"""

import logging
import mimetypes
from datetime import datetime
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.rag.models import FileInfo
from app.rag.object_store import object_name
from app.rag.parser import parse_file
from app.rag.splitter import split_overlap_paragraphs
from app.rag.vector import compatible_base_url

logger = logging.getLogger(__name__)
TEXT_TYPES = {"pdf", "docx", "doc", "txt"}
IMAGE_TYPES = {"jpg", "jpeg", "png", "gif", "bmp"}
LARGE_FILE_THRESHOLD = 5_000
IMAGE_PROMPT = (
    "请描述这张图片的内容，包括场景、对象、布局、颜色、文字信息，"
    "直接输出纯文本描述，不要多余说明，不要增加任何特殊符号，特别是换行符"
)


class FileService:
    def __init__(
        self,
        repository: object,
        objects: object,
        index: object | None,
        retriever: object | None,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._objects = objects
        self._index = index
        self._retriever = retriever
        self._settings = settings

    async def upload(self, file_name: str, content: bytes, content_type: str) -> FileInfo:
        """输入上传文件名/字节/MIME，输出可供旧前端使用的 FileInfo。

        需求是先保留原件，再让小文件直接问答、大文件走向量检索。中间状态
        PROCESSING 写 MySQL，失败写 FAILED；只有解析/嵌入步骤结束才标 SUCCESS，
        避免 Java 在解析前先写 SUCCESS 的并发窗口。向量化失败按 Java 降级为
        embed=0，不把上传整体判失败，但此时仅能问前 20,000 个展示字符。
        """
        if not content:
            raise ValueError("文件不能为空")
        if len(content) > self._settings.max_upload_bytes:
            raise ValueError("文件超过 50 MB 上传限制")
        file_type = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "unknown"
        info = FileInfo(
            file_id=str(uuid4()),
            file_name=file_name,
            file_type=file_type,
            file_size=len(content),
            created_at=datetime.now(),
            status="PROCESSING",
        )
        await self._repository.create(info)
        try:
            name = object_name(info.file_id, file_type)
            info.minio_path = await self._objects.put(name, content, content_type)
            await self._repository.update(info)
            if file_type in TEXT_TYPES:
                parsed = parse_file(content, file_type)
                info.extracted_text = parsed.truncated_text
                await self._repository.update(info)
                if len(parsed.full_text) >= LARGE_FILE_THRESHOLD:
                    try:
                        if self._index is None:
                            raise RuntimeError("未配置 EMBEDDING_API_KEY")
                        chunks = split_overlap_paragraphs(parsed.full_text)
                        await self._index.replace(info.file_id, chunks)
                        info.embed = 1
                    except Exception:
                        logger.exception("File indexing failed; upload falls back to display text")
            elif file_type in IMAGE_TYPES:
                info.extracted_text = await self._describe_image(content, file_type)
            info.status = "SUCCESS"
            await self._repository.update(info)
            return info
        except Exception:
            info.status = "FAILED"
            try:
                await self._repository.update(info)
            except Exception:
                logger.exception("Could not persist FAILED file status: %s", info.file_id)
            raise

    async def _describe_image(self, content: bytes, file_type: str) -> str:
        """输入图片字节，输出 Java 图片提示词生成的纯文本描述。"""
        import base64

        key = self._settings.vision_api_key or self._settings.openai_api_key
        if not key:
            raise RuntimeError("图片识别需要 VISION_API_KEY 或 OPENAI_API_KEY")
        model = ChatOpenAI(
            model=self._settings.vision_model,
            api_key=key,
            base_url=compatible_base_url(self._settings.vision_base_url),
            temperature=0.2,
        )
        mime = mimetypes.types_map.get(f".{file_type}", "image/png")
        data = base64.b64encode(content).decode("ascii")
        response = await model.ainvoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": IMAGE_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                    ]
                )
            ]
        )
        return str(response.content).strip() or "[无法识别图片内容]"

    async def get(self, file_id: str) -> FileInfo:
        """输入 fileId，输出已保存元数据；不存在时报告与 Java 相同的业务错误。"""
        info = await self._repository.get(file_id)
        if info is None:
            raise ValueError(f"文件不存在: {file_id}")
        return info

    async def content(self, file_id: str) -> str:
        """旧前端展示文本；只读 SUCCESS，返回 Java 的无内容提示。"""
        info = await self.get(file_id)
        if info.status != "SUCCESS":
            raise ValueError(f"文件尚未处理完成，当前状态: {info.status}")
        return info.extracted_text or "该文件没有可识别的内容"

    async def list_all(self) -> list[FileInfo]:
        return await self._repository.list_all()

    async def exists(self, file_id: str) -> bool:
        return await self._repository.get(file_id) is not None

    async def delete(self, file_id: str) -> None:
        """输入 fileId，先清向量和原件，最后清 MySQL；失败时不返回假成功。

        三种存储无法同一个事务回滚。这里保证删除步骤幂等；更严格的失败恢复
        需要上级设计中的 deletion job/worker，不能声称已实现跨库原子删除。
        """
        info = await self.get(file_id)
        if self._index is not None:
            await self._index.delete(file_id)
        elif info.embed:
            raise RuntimeError("向量库未配置，不能安全删除已向量化文件")
        if info.minio_path:
            await self._objects.delete(object_name(info.file_id, info.file_type))
        await self._repository.delete(file_id)

    async def load_content(self, file_id: str, question: str) -> str:
        """Agent 的 loadContent 业务结果，输入 fileId/问题，输出 Java 风格文本。

        embed=1 检索相关块；embed=0 直接用展示文本。工具结果附上文件名与类型，
        是为了让模型回答有上下文，但不把存储 URL 或其他文件内容交给模型。
        """
        try:
            info = await self.get(file_id)
            if info.status != "SUCCESS":
                return f"文件处理中或处理失败，当前状态: {info.status}，文件ID: {file_id}"
            if info.embed:
                if not question.strip():
                    body = "请提供具体问题以进行语义检索。"
                elif self._retriever is None:
                    body = "RAG 检索失败: 向量检索未配置"
                else:
                    try:
                        parts = await self._retriever.retrieve(file_id, question)
                        body = (
                            "相关内容: \n\n" + "".join(part + "\n\n" for part in parts)
                            if parts
                            else "未检索到与问题相关的内容"
                        )
                    except Exception:
                        logger.exception("RAG retrieval failed: %s", file_id)
                        body = "RAG 检索失败"
            else:
                body = await self.content(file_id)
            return (
                f"=== 文件信息 ===\n文件名: {info.file_name}\n"
                f"文件类型: {info.file_type}\n\n=== 文件内容 ===\n{body}"
            )
        except ValueError as exc:
            return str(exc)
        except Exception:
            logger.exception("loadContent failed: %s", file_id)
            return "加载文件内容失败"
