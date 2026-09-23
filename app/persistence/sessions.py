"""Async access to Java's existing ai_session table; never creates or drops tables."""

from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings


def mysql_async_url(settings: Settings) -> URL:
    raw = settings.mysql_url.strip()
    if raw.startswith("jdbc:mysql://"):
        parsed = urlsplit(raw.removeprefix("jdbc:"))
        database = parsed.path.lstrip("/")
        if not parsed.hostname or not database:
            raise ValueError("MYSQL_URL 必须包含主机和数据库名")
        return URL.create(
            "mysql+asyncmy",
            username=settings.mysql_username,
            password=settings.mysql_password,
            host=parsed.hostname,
            port=parsed.port or 3306,
            database=database,
            query={"charset": "utf8mb4"},
        )
    url = make_url(raw)
    if url.drivername != "mysql+asyncmy":
        raise ValueError("MYSQL_URL 必须是 Java JDBC URL 或 mysql+asyncmy URL")
    return url


def create_mysql_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        mysql_async_url(settings),
        pool_size=5,
        max_overflow=15,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
    )


class MySQLSessionRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def recent(self, conversation_id: str, limit: int = 30) -> list[tuple[str, str]]:
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT question, answer FROM ai_session "
                    "WHERE session_id = :session_id AND question IS NOT NULL "
                    "ORDER BY id DESC LIMIT :limit"
                ),
                {"session_id": conversation_id, "limit": limit},
            )
            rows = result.all()
        return [(row.question, row.answer or "") for row in reversed(rows)]

    async def recent_file(
        self, conversation_id: str, file_id: str, limit: int = 30
    ) -> list[tuple[str, str]]:
        """文件问答只读取同一 fileId 的历史，避免切换附件时混入别的文件答案。"""
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT question, answer FROM ai_session "
                    "WHERE session_id=:session_id AND fileid=:file_id "
                    "AND question IS NOT NULL ORDER BY id DESC LIMIT :limit"
                ),
                {"session_id": conversation_id, "file_id": file_id, "limit": limit},
            )
            rows = result.all()
        return [(row.question, row.answer or "") for row in reversed(rows)]

    async def file_ids_for_conversation(self, conversation_id: str) -> list[str]:
        """找出 MySQL 中明确关联此会话的文件，交给 FileService 做跨库清理。"""
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text("SELECT file_id FROM ai_file_info WHERE conversation_id=:session_id"),
                {"session_id": conversation_id},
            )
            return [row.file_id for row in result.all()]

    async def create(
        self,
        conversation_id: str,
        question: str,
        *,
        agent_type: str | None = "chat",
        file_id: str | None = None,
    ) -> int:
        """创建一条待回填会话；agent_type 可为 None 以兼容 Java 未设置该字段的路径。"""
        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    "INSERT INTO ai_session (session_id, question, agent_type, fileid) "
                    "VALUES (:session_id, :question, :agent_type, :file_id)"
                ),
                {
                    "session_id": conversation_id,
                    "question": question,
                    "agent_type": agent_type,
                    "file_id": file_id,
                },
            )
            return int(result.lastrowid)

    async def finish(
        self,
        record_id: int,
        *,
        answer: str,
        thinking: str,
        tools: str,
        reference: str | None,
        recommend: str | None,
        first_response_time: int | None,
        total_response_time: int,
    ) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE ai_session SET answer = :answer, thinking = :thinking, "
                    "tools = :tools, `reference` = :reference, recommend = :recommend, "
                    "first_response_time = :first_response_time, "
                    "total_response_time = :total_response_time "
                    "WHERE id = :id"
                ),
                {
                    "id": record_id,
                    "answer": answer,
                    "thinking": thinking,
                    "tools": tools,
                    "reference": reference,
                    "recommend": recommend,
                    "first_response_time": first_response_time,
                    "total_response_time": total_response_time,
                },
            )

    async def list_page(self, page_num: int, page_size: int) -> dict[str, Any]:
        async with self._engine.connect() as connection:
            total = await connection.scalar(
                text("SELECT COUNT(DISTINCT session_id) FROM ai_session")
            )
            result = await connection.execute(
                text(
                    "SELECT head.session_id AS conversationId, head.agent_type AS agentType, "
                    "head.question, head.answer, grouped.message_count AS messageCount, "
                    "head.create_time AS createTime, grouped.last_update AS updateTime, "
                    "head.fileid FROM ai_session AS head "
                    "JOIN (SELECT session_id, MIN(id) AS first_id, COUNT(*) AS message_count, "
                    "MAX(update_time) AS last_update FROM ai_session "
                    "GROUP BY session_id) AS grouped "
                    "ON head.id = grouped.first_id "
                    "ORDER BY grouped.last_update DESC, head.id DESC "
                    "LIMIT :limit OFFSET :offset"
                ),
                {"limit": page_size, "offset": (page_num - 1) * page_size},
            )
            records = [dict(row) for row in result.mappings().all()]
        return {"pageNum": page_num, "pageSize": page_size, "total": total or 0, "records": records}

    async def detail(self, conversation_id: str) -> dict[str, Any] | None:
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT id, question, answer, thinking, tools, `reference` AS `reference`, "
                    "create_time AS createTime, fileid, recommend, agent_type AS agentType "
                    "FROM ai_session WHERE session_id = :session_id ORDER BY id ASC"
                ),
                {"session_id": conversation_id},
            )
            rows = [dict(row) for row in result.mappings().all()]
        if not rows:
            return None
        agent_type = rows[0].pop("agentType")
        for row in rows[1:]:
            row.pop("agentType")
        return {
            "conversationId": conversation_id,
            "agentType": agent_type,
            "fileid": rows[0]["fileid"],
            "messages": rows,
        }

    async def delete(self, conversation_id: str) -> bool:
        # Match Java's transactional delete of associated records and session rows.
        async with self._engine.begin() as connection:
            exists = await connection.scalar(
                text("SELECT id FROM ai_session WHERE session_id = :session_id LIMIT 1"),
                {"session_id": conversation_id},
            )
            if exists is None:
                return False
            for table in ("ai_file_info", "ai_ppt_inst"):
                await connection.execute(
                    text(f"DELETE FROM {table} WHERE conversation_id = :session_id"),
                    {"session_id": conversation_id},
                )
            await connection.execute(
                text("DELETE FROM ai_session WHERE session_id = :session_id"),
                {"session_id": conversation_id},
            )
        return True
