# Dodo Agent Python

这是 Dodo Agent 的 Python/LangGraph 渐进式迁移项目。当前阶段已经实现：

- FastAPI 应用与健康检查；
- 与旧前端兼容的 SSE 消息协议；
- 基于 LangChain `create_agent` 的通用 ReAct Factory；
- 通用 ReAct Runner、LangGraph 会话内存和工具事件适配；
- DeepSeek/OpenAI-compatible 模型接入；
- 一个安全的当前时间工具；
- 未配置模型密钥时的稳定错误事件；
- 独立配置与测试；
- 可选择直接托管旧项目的静态前端。
- 按 Java 行为迁移的 Tavily MCP 联网搜索首版（需要模型和 Tavily 密钥）。
- 会话记录、停止正在运行的任务；
- 文件上传、内容解析、图片描述和 RAG 问答。
- Java 兼容的深度研究首版：需求澄清、研究主题、三轮 Plan-Execute、并行搜索、评审、报告、来源、会话保存与停止。

## 架构设计

`app/api` 提供 HTTP/SSE 接口；`app/web_search` 和 `app/deep_research` 分别实现联网搜索与深度研究；`app/rag` 负责文件处理和检索；`app/runtime`、`app/persistence` 提供运行时与会话存储能力。本地研究文档不随 GitHub 仓库发布。

## 安装

```powershell
conda create -n dodo-langgraph python=3.12 -y
conda activate dodo-langgraph
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 配置

复制 `.env.example` 为 `.env`，按使用的功能填写自己的连接信息和密钥：

```powershell
Copy-Item .env.example .env
```

| 功能 | 需要配置的环境变量 |
| --- | --- |
| 模型调用 | `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`CHAT_MODEL` |
| 联网搜索与深度研究 | `TAVILY_API_KEY`、`TAVILY_MCP_URL` |
| 会话与文件元数据 | `MYSQL_URL`、`MYSQL_USERNAME`、`MYSQL_PASSWORD` |
| 跨实例任务管理 | `REDIS_HOST`、`REDIS_PORT`、`REDIS_DATABASE`、`REDIS_PASSWORD` |
| 文件存储 | `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET_NAME` |
| 大文件向量检索 | `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`，以及 `PGVECTOR_URL`，或 `PGVECTOR_HOST`、`PGVECTOR_PORT`、`PGVECTOR_DATABASE`、`PGVECTOR_USER`、`PGVECTOR_PASSWORD` |
| 图片识别 | `VISION_BASE_URL`、`VISION_MODEL`；`VISION_API_KEY` 可单独配置，留空时沿用 `OPENAI_API_KEY` |

按实际使用的功能启动对应服务。不要提交包含真实密钥的 `.env`；其他可选项见 `.env.example`。

## 启动

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8888
```

可用入口：

- 健康检查：`http://127.0.0.1:8888/health`
- OpenAPI：`http://127.0.0.1:8888/docs`
- SSE 基线：`http://127.0.0.1:8888/agent/chat/stream?query=hello&conversationId=test`
- 深度研究：`http://127.0.0.1:8888/agent/deep/stream?query=hello&conversationId=deep-test`
- 停止当前会话：`http://127.0.0.1:8888/agent/stop?conversationId=test`
- 会话列表：`http://127.0.0.1:8888/session/list?pageNum=1&pageSize=10`
- 会话详情：`http://127.0.0.1:8888/session/test`

当 `SERVE_FRONTEND=true` 时，服务会尝试托管旧项目
`../Agent/src/main/resources/static` 下的前端资源。
仅克隆本 Python 仓库不会包含这些前端文件；如需使用旧前端，需另外获取 Java 项目并按上述相邻目录结构放置。

## 测试

```powershell
pytest
ruff check .
```
