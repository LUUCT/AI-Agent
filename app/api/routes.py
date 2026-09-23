import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.persistence.sessions import MySQLSessionRepository
from app.runtime.runner import AgentStreamRunner

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "dodo-agent-python"}


def get_chat_runner(request: Request) -> AgentStreamRunner:
    """从应用状态取当前 Agent；启动阶段可能将通用 Agent 替换成联网搜索 Agent。"""
    return request.app.state.chat_runner


def get_deep_runner(request: Request) -> AgentStreamRunner:
    """从应用状态取得深度研究运行器；未配置时返回启动阶段放入的错误运行器。"""
    return request.app.state.deep_runner

def get_session_repository(request: Request) -> MySQLSessionRepository:
    """返回共享的 MySQL 会话仓储，供列表、详情和删除接口使用。"""
    return request.app.state.session_repository


def _success(data: Any) -> dict[str, Any]:
    return {"code": 200, "message": "", "data": data}


def _failure(message: str) -> dict[str, Any]:
    return {"code": 500, "message": message, "data": None}


@router.get("/session/list")
async def session_list(
    repository: Annotated[MySQLSessionRepository, Depends(get_session_repository)],
    page_num: Annotated[int, Query(alias="pageNum", ge=1)] = 1,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 10,
) -> dict[str, Any]:
    try:
        return _success(await repository.list_page(page_num, page_size))
    except Exception:
        logger.exception("Failed to list sessions")
        return _failure("查询会话列表失败")


@router.get("/session/{conversation_id}")
async def session_detail(
    conversation_id: str,
    repository: Annotated[MySQLSessionRepository, Depends(get_session_repository)],
) -> dict[str, Any]:
    try:
        detail = await repository.detail(conversation_id)
        return _success(detail) if detail is not None else _failure("会话不存在")
    except Exception:
        logger.exception("Failed to load session: conversation_id=%s", conversation_id)
        return _failure("查询会话详情失败")


@router.delete("/session/{conversation_id}")
async def delete_session(
    conversation_id: str,
    repository: Annotated[MySQLSessionRepository, Depends(get_session_repository)],
    request: Request,
) -> dict[str, Any]:
    try:
        if await repository.detail(conversation_id) is None:
            return _failure("会话不存在")
        # 旧代码直接删 ai_file_info 行，MinIO/PGVector 会残留。先调用同一文件删除
        # 服务清理明确关联该会话的文件；任何一步失败时，不向前端报告删除成功。
        if hasattr(repository, "file_ids_for_conversation"):
            for file_id in await repository.file_ids_for_conversation(conversation_id):
                await request.app.state.file_service.delete(file_id)
        return (
            {"code": 200, "message": "会话删除成功", "data": None}
            if await repository.delete(conversation_id)
            else _failure("会话不存在")
        )
    except Exception:
        logger.exception("Failed to delete session: conversation_id=%s", conversation_id)
        return _failure("删除会话失败")


@router.get("/agent/chat/stream")
async def chat_stream(
    query: Annotated[str, Query(min_length=1)],
    conversation_id: Annotated[str, Query(alias="conversationId", min_length=1)],
    runner: Annotated[AgentStreamRunner, Depends(get_chat_runner)],
) -> StreamingResponse:
    """旧前端聊天入口；输入为问题和会话 ID，输出为逐条发送的 SSE 事件流。"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="查询参数不能为空")

    async def event_stream():
        # runner 生成业务事件；这里仅转换成前端可读取的 data: JSON 格式。
        async for event in runner.stream(query, conversation_id):
            yield event.to_sse()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/agent/deep/stream")
async def deep_stream(
    query: Annotated[str, Query(min_length=1)],
    conversation_id: Annotated[str, Query(alias="conversationId", min_length=1)],
    runner: Annotated[AgentStreamRunner, Depends(get_deep_runner)],
) -> StreamingResponse:
    """Java 兼容的深度研究入口；输入问题和会话 ID，输出研究阶段及报告事件。"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="查询参数不能为空")

    async def event_stream():
        """把运行器产生的领域事件逐条编码为旧前端接受的 SSE data 消息。"""
        async for event in runner.stream(query, conversation_id):
            yield event.to_sse()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.get("/agent/stop")
async def stop_agent(
    conversation_id: Annotated[str, Query(alias="conversationId", min_length=1)],
    runner: Annotated[AgentStreamRunner, Depends(get_chat_runner)],
    request: Request,
) -> dict[str, str | bool]:
    """按会话 ID 停止执行，返回旧前端约定的 success/message。"""
    stopped = await runner.stop(conversation_id) if hasattr(runner, "stop") else False
    file_runner = getattr(request.app.state, "file_runner", None)
    if not stopped and file_runner is not None:
        stopped = await file_runner.stop(conversation_id)
    deep_runner = getattr(request.app.state, "deep_runner", None)
    if not stopped and deep_runner is not None and hasattr(deep_runner, "stop"):
        stopped = await deep_runner.stop(conversation_id)
    return {
        "success": stopped,
        "message": "已停止执行" if stopped else "没有找到正在执行的任务或已停止",
    }
