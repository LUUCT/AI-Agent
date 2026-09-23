"""旧 Java 文件 HTTP/SSE 路由，字段名与 BaseResult 形状保持一致。"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.models.events import StreamEvent, StreamEventType
from app.rag.service import FileService

router = APIRouter()
logger = logging.getLogger(__name__)


def get_files(request: Request) -> FileService:
    return request.app.state.file_service


def success(data: Any, message: str = "") -> dict[str, Any]:
    return {"code": 200, "message": message, "data": data}


def failure(message: str) -> dict[str, Any]:
    return {"code": 500, "message": message, "data": None}


@router.post("/file/upload")
async def upload_file(
    file: Annotated[UploadFile, File()],
    files: Annotated[FileService, Depends(get_files)],
) -> dict[str, Any]:
    """输入 multipart file，输出 Java FileInfo；只读取上限+1字节来检查 50 MB。"""
    try:
        content = await file.read(files._settings.max_upload_bytes + 1)
        if not content:
            return failure("文件不能为空")
        info = await files.upload(file.filename or "unknown", content, file.content_type or "")
        return success(info.to_legacy_dict())
    except Exception as exc:
        logger.exception("File upload failed")
        return failure(f"文件上传失败: {exc}")
    finally:
        await file.close()


@router.get("/file/info/{file_id}")
async def file_info(file_id: str, files: Annotated[FileService, Depends(get_files)]) -> dict:
    try:
        return success((await files.get(file_id)).to_legacy_dict())
    except Exception as exc:
        return failure(f"获取文件信息失败: {exc}")


@router.get("/file/content/{file_id}")
async def file_content(file_id: str, files: Annotated[FileService, Depends(get_files)]) -> dict:
    try:
        content = await files.content(file_id)
        return success({"content": content, "length": len(content)})
    except Exception as exc:
        return failure(f"获取文件内容失败: {exc}")


@router.get("/file/list")
async def file_list(files: Annotated[FileService, Depends(get_files)]) -> dict:
    try:
        infos = await files.list_all()
        return success(
            {"count": len(infos), "files": {i.file_id: i.to_legacy_dict() for i in infos}}
        )
    except Exception as exc:
        return failure(f"获取文件列表失败: {exc}")


@router.get("/file/exists/{file_id}")
async def file_exists(file_id: str, files: Annotated[FileService, Depends(get_files)]) -> dict:
    try:
        return success(await files.exists(file_id))
    except Exception as exc:
        return failure(f"检查文件存在失败: {exc}")


@router.delete("/file/{file_id}")
async def file_delete(file_id: str, files: Annotated[FileService, Depends(get_files)]) -> dict:
    try:
        await files.delete(file_id)
        return success(None, "文件删除成功")
    except Exception as exc:
        return failure(f"删除文件失败: {exc}")


@router.get("/agent/file/stream")
async def file_stream(
    request: Request,
    query: Annotated[str, Query(min_length=1)],
    conversation_id: Annotated[str, Query(alias="conversationId", min_length=1)],
    file_id: Annotated[str, Query(alias="fileId", min_length=1)],
) -> StreamingResponse:
    """旧前端的三个 query 参数不变；业务事件转换为 data: JSON 的 SSE。"""

    async def events():
        runner = request.app.state.file_runner
        if runner is None:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="文件问答模型未配置，请设置 OPENAI_API_KEY。",
                code="FILE_MODEL_UNAVAILABLE",
            ).to_sse()
            return
        async for event in runner.stream(query, conversation_id, file_id):
            yield event.to_sse()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
