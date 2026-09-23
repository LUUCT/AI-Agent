import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis

from app.agents.chat import build_chat_model, build_chat_runner
from app.api.file_routes import router as file_router
from app.api.routes import router
from app.core.config import get_settings
from app.deep_research import DeepResearchRunner
from app.persistence.sessions import MySQLSessionRepository, create_mysql_engine
from app.rag.object_store import MinioObjectStore
from app.rag.repository import MySQLFileRepository
from app.rag.retriever import RagRetriever
from app.rag.service import FileService
from app.rag.vector import PgVectorIndex
from app.runtime.file_chat import FileChatRunner
from app.runtime.runner import UnavailableAgentRunner
from app.runtime.task_manager import AgentTaskManager
from app.web_search import WebSearchRunner
from app.web_search.mcp import tavily_tools

settings = get_settings()
logger = logging.getLogger(__name__)
# 启动时创建连接池和仓储对象；具体 SQL 只在请求处理时执行。
mysql_engine = create_mysql_engine(settings)
session_repository = MySQLSessionRepository(mysql_engine)
file_repository = MySQLFileRepository(mysql_engine)
object_store = MinioObjectStore(settings)
# 仅在嵌入和 PGVector 配置齐全时连接向量库；小文件直读仍可独立使用。
try:
    vector_index = PgVectorIndex(settings) if settings.embedding_api_key.strip() else None
except Exception:
    logger.exception("RAG vector index initialization failed")
    vector_index = None
rag_retriever = (
    RagRetriever(build_chat_model(settings), vector_index)
    if vector_index is not None and settings.openai_api_key.strip()
    else None
)
file_service = FileService(file_repository, object_store, vector_index, rag_retriever, settings)
# 对应 Java AgentTaskManager：Redis 负责跨实例任务互斥、停止通知和租约续期。
task_manager = AgentTaskManager(
    Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_database,
        password=settings.redis_password or None,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """启动文件、搜索和研究 Agent，关闭时释放 Redis、MySQL 与 PGVector 连接。"""
    try:
        async with AsyncExitStack() as stack:
            if vector_index is not None:
                stack.push_async_callback(vector_index.close)
            manager = None
            if settings.openai_api_key.strip():
                try:
                    await task_manager.start()
                    manager = task_manager
                    stack.push_async_callback(task_manager.close)
                except Exception:
                    logger.exception(
                        "Redis task manager unavailable; file Agent uses local cancellation"
                    )
                model = build_chat_model(settings)
                application.state.file_runner = FileChatRunner(
                    model,
                    file_service,
                    session_repository,
                    manager,
                    max_rounds=settings.agent_recursion_limit,
                )
            if settings.openai_api_key.strip() and settings.tavily_api_key.strip():
                try:
                    tools = await stack.enter_async_context(tavily_tools(settings))
                    application.state.chat_runner = WebSearchRunner(
                        model, tools, session_repository, manager
                    )
                    application.state.deep_runner = DeepResearchRunner(
                        model, tools, session_repository, manager
                    )
                except Exception:
                    logger.exception("Tavily MCP initialization failed")
                    application.state.chat_runner = UnavailableAgentRunner(
                        "Tavily MCP 初始化失败，请检查相关配置。"
                    )
                    application.state.deep_runner = UnavailableAgentRunner(
                        "Tavily MCP 初始化失败，请检查相关配置。"
                    )
            yield
    finally:
        await mysql_engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Python/LangGraph migration baseline for Dodo Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins(),
    allow_credentials=settings.cors_origins != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# 默认使用通用 ReAct；搜索服务连接成功后，lifespan 才会替换为联网搜索运行器。
app.state.chat_runner = build_chat_runner(settings)
app.state.deep_runner = UnavailableAgentRunner("深度研究需要配置模型与 Tavily MCP。")
app.state.session_repository = session_repository
app.state.file_service = file_service
app.state.file_runner = (
    FileChatRunner(build_chat_model(settings), file_service, session_repository)
    if settings.openai_api_key.strip()
    else None
)
app.include_router(router)
app.include_router(file_router)

repository_root = Path(__file__).resolve().parents[2]
legacy_frontend = repository_root / "Agent" / "src" / "main" / "resources" / "static"

if settings.serve_frontend and legacy_frontend.is_dir():
    app.mount("/", StaticFiles(directory=legacy_frontend, html=True), name="frontend")
