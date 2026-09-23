"""对应 Java FileInfo 与 ai_file_info 表的领域对象。"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class FileInfo:
    """文件元数据。

    输入来自上传请求或 MySQL 查询；输出由 ``to_legacy_dict`` 转成旧前端使用的
    camelCase JSON。把转换集中在这里，可防止 API 各处字段名不一致。
    """

    file_id: str
    file_name: str
    file_type: str
    file_size: int
    minio_path: str | None = None
    extracted_text: str | None = None
    created_at: datetime | None = None
    conversation_id: str | None = None
    status: str = "PENDING"
    embed: int = 0

    def to_legacy_dict(self) -> dict[str, object]:
        """输出 Java ``FileInfo`` 的字段，让现有网页不必改名。"""
        return {
            "fileId": self.file_id,
            "fileName": self.file_name,
            "fileType": self.file_type,
            "fileSize": self.file_size,
            "minioPath": self.minio_path,
            "extractedText": self.extracted_text,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "conversationId": self.conversation_id,
            "status": self.status,
            "embed": self.embed,
            "processed": self.status == "SUCCESS" and self.extracted_text is not None,
            "image": self.file_type in {"png", "jpg", "jpeg", "gif", "bmp"},
            "pdf": self.file_type == "pdf",
            "word": self.file_type in {"doc", "docx"},
        }
