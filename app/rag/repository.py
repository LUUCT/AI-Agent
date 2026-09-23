"""复用 Java 的 ai_file_info MySQL 表，不自动新建或删除业务表。"""

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.rag.models import FileInfo


class MySQLFileRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @staticmethod
    def _from_row(row: dict) -> FileInfo:
        """把数据库下划线列名变成业务对象，避免 SQL 行结构流入 Agent。"""
        return FileInfo(
            file_id=row["file_id"],
            file_name=row["file_name"],
            file_type=row["file_type"] or "unknown",
            file_size=row["file_size"] or 0,
            minio_path=row["minio_path"],
            extracted_text=row["extracted_text"],
            created_at=row["created_at"],
            conversation_id=row["conversation_id"],
            status=row["status"] or "PENDING",
            embed=row["embed"] or 0,
        )

    async def create(self, info: FileInfo) -> None:
        """上传开始即保存 PROCESSING；输入 FileInfo，输出持久化记录。"""
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO ai_file_info (file_id,file_name,file_type,file_size,"
                    "created_at,status,embed) VALUES (:file_id,:file_name,:file_type,"
                    ":file_size,:created_at,:status,:embed)"
                ),
                {
                    "file_id": info.file_id,
                    "file_name": info.file_name,
                    "file_type": info.file_type,
                    "file_size": info.file_size,
                    "created_at": info.created_at or datetime.now(),
                    "status": info.status,
                    "embed": info.embed,
                },
            )

    async def update(self, info: FileInfo) -> None:
        """写入上传/解析/嵌入的中间结果，供状态查询和后续问答使用。"""
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE ai_file_info SET minio_path=:minio_path, "
                    "extracted_text=:extracted_text, status=:status, embed=:embed, "
                    "update_time=NOW() WHERE file_id=:file_id"
                ),
                {
                    "file_id": info.file_id,
                    "minio_path": info.minio_path,
                    "extracted_text": info.extracted_text,
                    "status": info.status,
                    "embed": info.embed,
                },
            )

    async def get(self, file_id: str) -> FileInfo | None:
        """按服务端文件 ID 查一条；不存在时返回 None。"""
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text("SELECT * FROM ai_file_info WHERE file_id=:file_id LIMIT 1"),
                {"file_id": file_id},
            )
            row = result.mappings().first()
        return self._from_row(dict(row)) if row is not None else None

    async def list_all(self) -> list[FileInfo]:
        """保持 Java 列表接口的全部文件语义；大规模分页留给新版接口。"""
        async with self._engine.connect() as connection:
            result = await connection.execute(text("SELECT * FROM ai_file_info ORDER BY id DESC"))
            return [self._from_row(dict(row)) for row in result.mappings().all()]

    async def delete(self, file_id: str) -> None:
        """对象和向量清理成功后，最后移除 MySQL 元数据。"""
        async with self._engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM ai_file_info WHERE file_id=:file_id"), {"file_id": file_id}
            )
