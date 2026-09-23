"""Java FileParserService 的 PDF、DOCX、TXT 提取与展示截断。"""

from dataclasses import dataclass
from io import BytesIO

TRUNCATION_SUFFIX = "\n\n... (内容已截断，文件过长)"
MAX_TEXT_LENGTH = 20_000


@dataclass(frozen=True)
class ParseResult:
    full_text: str
    truncated_text: str


def parse_file(content: bytes, file_type: str) -> ParseResult:
    """从上传字节提取文本。

    输入为原始文件字节及文件扩展名，输出同时保留全文和最多 20,000 字的展示版。
    全文仅用于 5,000 字阈值判断与向量化；展示版写入 MySQL，避免大文本直接
    塞进会话或旧前端。PDF/DOCX 依赖在函数内部导入，使其他接口可独立启动。
    """
    if file_type == "txt":
        full_text = content.decode("utf-8").strip()
    elif file_type == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    elif file_type == "docx":
        from docx import Document

        document = Document(BytesIO(content))
        full_text = "\n".join(
            paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
        ).strip()
    elif file_type == "doc":
        raise ValueError("暂不支持 .doc 格式，请转换为 .docx")
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")
    truncated = (
        full_text[:MAX_TEXT_LENGTH] + TRUNCATION_SUFFIX
        if len(full_text) > MAX_TEXT_LENGTH
        else full_text
    )
    return ParseResult(full_text=full_text, truncated_text=truncated)
