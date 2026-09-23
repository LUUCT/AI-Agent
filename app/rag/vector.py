"""LangChain 嵌入模型 + 独立 PGVector 表；不读写 Spring AI 的旧向量表。"""

from uuid import NAMESPACE_URL, uuid5

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings

EMBEDDING_BATCH_SIZE = 9


def compatible_base_url(url: str) -> str:
    """OpenAI SDK 要求 API 根路径；Java 的 DashScope 配置只给 compatible-mode。"""
    clean = url.rstrip("/")
    return f"{clean}/v1" if clean.endswith("/compatible-mode") else clean


class PgVectorIndex:
    def __init__(self, settings: Settings) -> None:
        """连接独立向量库，并使用 Java 同款 1024 维嵌入模型。"""
        if not settings.embedding_api_key:
            raise ValueError("请配置 EMBEDDING_API_KEY")
        pgvector_url = settings.resolved_pgvector_url()
        if settings.embedding_dimensions != 1024:
            raise ValueError("Java RAG 兼容模式要求 EMBEDDING_DIMENSIONS=1024")
        self._engine: AsyncEngine = create_async_engine(
            pgvector_url, pool_pre_ping=True
        )
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=compatible_base_url(settings.embedding_base_url),
            dimensions=settings.embedding_dimensions,
        )

    async def close(self) -> None:
        await self._engine.dispose()

    async def ensure_schema(self) -> None:
        """只创建 Python 专用表与索引，永不修改 Java 的 vector_file_info。

        首次写入前执行；PostgreSQL 用户需能安装 vector 扩展。独立表解决
        Spring AI 与 Python 向量存储 schema 不兼容的问题。
        """
        async with self._engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS dodo_rag_chunks ("
                    "id uuid PRIMARY KEY, file_id varchar(255) NOT NULL, "
                    "chunk_index integer NOT NULL, content text NOT NULL, "
                    "embedding vector(1024) NOT NULL, "
                    "UNIQUE (file_id, chunk_index))"
                )
            )
            await connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_dodo_rag_file "
                    "ON dodo_rag_chunks (file_id)"
                )
            )
            await connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_dodo_rag_cosine "
                    "ON dodo_rag_chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )

    async def replace(self, file_id: str, chunks: list[str]) -> None:
        """输入 fileId 和切块；输出为该文件完整的新向量集。

        先按 Java 的每批 9 块调用嵌入模型，再在一个 PostgreSQL 事务内替换
        旧块。所有向量都拿到后才写库，避免模型在第 N 批失败留下半套索引。
        """
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            vectors.extend(await self._embeddings.aembed_documents(chunks[start : start + 9]))
        if len(vectors) != len(chunks) or any(len(v) != 1024 for v in vectors):
            raise ValueError("嵌入向量数量或维度与 Java 配置不一致")
        await self.ensure_schema()
        async with self._engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM dodo_rag_chunks WHERE file_id=:file_id"),
                {"file_id": file_id},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                chunk_id = uuid5(NAMESPACE_URL, f"dodo-rag:{file_id}:{index}")
                await connection.execute(
                    text(
                        "INSERT INTO dodo_rag_chunks "
                        "(id,file_id,chunk_index,content,embedding) VALUES "
                        "(:id,:file_id,:chunk_index,:content,CAST(:embedding AS vector))"
                    ),
                    {
                        "id": chunk_id,
                        "file_id": file_id,
                        "chunk_index": index,
                        "content": chunk,
                        "embedding": str(vector),
                    },
                )

    async def search(self, file_id: str, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        """只搜索指定 fileId，返回 (稳定块 ID, 文本) 供多查询结果去重。"""
        vector = await self._embeddings.aembed_query(query)
        if len(vector) != 1024:
            raise ValueError("查询向量维度不是 1024")
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT id, content FROM dodo_rag_chunks "
                    "WHERE file_id=:file_id "
                    "ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT :top_k"
                ),
                {"file_id": file_id, "embedding": str(vector), "top_k": top_k},
            )
            return [(str(row.id), row.content) for row in result.all()]

    async def delete(self, file_id: str) -> None:
        """按 fileId 删除所有块，修复 Java 只删原件而遗留向量的问题。"""
        await self.ensure_schema()
        async with self._engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM dodo_rag_chunks WHERE file_id=:file_id"),
                {"file_id": file_id},
            )
