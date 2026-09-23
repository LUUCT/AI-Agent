from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool

from app.core.config import Settings


@asynccontextmanager
async def tavily_tools(settings: Settings) -> AsyncIterator[list[BaseTool]]:
    """连接 Tavily MCP 并动态获取工具；输入为配置，输出为可供模型调用的工具列表。"""
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport
    from langchain.mcp import MCPAdapter

    # 与 Java 相同，走 MCP Streamable HTTP；密钥用于连接时的认证。
    transport = StreamableHttpTransport(
        settings.tavily_mcp_url,
        auth=settings.tavily_api_key,
    )
    client = Client(transport, timeout=settings.tavily_request_timeout_seconds)
    # 上下文管理器在搜索运行期间保持 MCP 连接，并在应用关闭时释放资源。
    async with MCPAdapter(client) as adapter:
        # list_tools 从服务端读取真实工具名和参数 schema，避免把 Tavily 接口写死。
        tools = await adapter.list_tools()
        if not tools:
            raise RuntimeError("Tavily MCP 没有返回可用工具")
        yield tools
