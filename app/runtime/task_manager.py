"""对应 Java AgentTaskManager：记录运行中的会话，并支持跨实例停止。"""

import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """本实例持有的 asyncio 任务及 Agent 类型；Redis 只保存实例归属。"""

    task: asyncio.Task
    agent_type: str


class AgentTaskManager:
    """用本地任务表加 Redis 租约实现会话互斥、停止通知和自动过期。"""

    task_key_prefix = "agent:task:"
    stop_topic_name = "agent:stop"
    task_ttl_seconds = 30 * 60
    refresh_interval_seconds = 5 * 60

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client
        self.instance_id = uuid4().hex[:8]
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._pubsub = None
        self._listener_task: asyncio.Task | None = None
        self._refresh_task: asyncio.Task | None = None

    def _key(self, conversation_id: str) -> str:
        """输入会话 ID，输出对应 Redis 任务键。"""
        return f"{self.task_key_prefix}{conversation_id}"

    async def start(self) -> None:
        """验证 Redis、订阅停止频道，并启动停止监听和租约续期任务。"""
        await self._redis.ping()
        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(self.stop_topic_name)
        self._listener_task = asyncio.create_task(self._listen_for_stops())
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def close(self) -> None:
        """应用关闭时取消任务、退订频道、释放本实例租约和 Redis 连接。"""
        for info in list(self._tasks.values()):
            info.task.cancel()
        for background in (self._listener_task, self._refresh_task):
            if background is not None:
                background.cancel()
        await asyncio.gather(
            *(task for task in (self._listener_task, self._refresh_task) if task is not None),
            return_exceptions=True,
        )
        if self._pubsub is not None:
            try:
                await self._pubsub.unsubscribe(self.stop_topic_name)
            except Exception:
                logger.exception("Failed to unsubscribe Agent stop topic")
            finally:
                await self._pubsub.aclose()
        for conversation_id, info in list(self._tasks.items()):
            try:
                await self.unregister_task(conversation_id, info.task)
            except Exception:
                logger.exception("Failed to remove task on shutdown: %s", conversation_id)
        await self._redis.aclose()

    async def register_task(
        self, conversation_id: str, task: asyncio.Task, agent_type: str
    ) -> bool:
        """登记会话任务；本地或 Redis 已占用则返回 False，成功返回 True。"""
        async with self._lock:
            if conversation_id in self._tasks:
                return False
            # Redis SET NX EX 是跨实例互斥锁；EX 防止进程异常退出后永久占用。
            acquired = await self._redis.set(
                self._key(conversation_id),
                self.instance_id,
                nx=True,
                ex=self.task_ttl_seconds,
            )
            if not acquired:
                return False
            self._tasks[conversation_id] = TaskInfo(task=task, agent_type=agent_type)
            return True

    async def unregister_task(self, conversation_id: str, task: asyncio.Task) -> None:
        """移除指定任务；仅删除仍属于当前实例的 Redis 键，避免误删新任务。"""
        async with self._lock:
            info = self._tasks.get(conversation_id)
            if info is None or info.task is not task:
                return
            del self._tasks[conversation_id]
            key = self._key(conversation_id)
            if await self._redis.get(key) == self.instance_id:
                await self._redis.delete(key)

    async def has_running_task(self, conversation_id: str) -> bool:
        """查询某会话是否在本实例或其他实例上运行。"""
        if conversation_id in self._tasks:
            return True
        return bool(await self._redis.exists(self._key(conversation_id)))

    async def stop_task(self, conversation_id: str) -> bool:
        """本地任务直接取消；其他实例的任务通过 Redis 发布停止消息。"""
        info = self._tasks.get(conversation_id)
        if info is not None:
            info.task.cancel()
            return True
        holder = await self._redis.get(self._key(conversation_id))
        if holder is None or holder == self.instance_id:
            return False
        await self._redis.publish(self.stop_topic_name, conversation_id)
        return True

    async def _listen_for_stops(self) -> None:
        """消费跨实例停止消息；收到本实例负责的会话 ID 时取消对应任务。"""
        try:
            async for message in self._pubsub.listen():
                if message.get("type") != "message":
                    continue
                data = message.get("data")
                conversation_id = data.decode() if isinstance(data, bytes) else str(data)
                info = self._tasks.get(conversation_id)
                if info is not None:
                    info.task.cancel()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Redis Agent stop listener failed")

    async def _refresh_loop(self) -> None:
        """后台定期刷新仍在运行的任务租约，防止长搜索在 30 分钟后失锁。"""
        try:
            while True:
                await asyncio.sleep(self.refresh_interval_seconds)
                await self.refresh_task_ttls()
        except asyncio.CancelledError:
            raise

    async def refresh_task_ttls(self) -> None:
        """只续期归当前实例所有的键；归属变化时清理本地任务登记。"""
        for conversation_id, info in list(self._tasks.items()):
            try:
                key = self._key(conversation_id)
                if await self._redis.get(key) == self.instance_id:
                    await self._redis.expire(key, self.task_ttl_seconds)
                elif self._tasks.get(conversation_id) is info:
                    logger.warning("Task lease changed: conversation_id=%s", conversation_id)
                    del self._tasks[conversation_id]
            except Exception:
                logger.exception("Failed to refresh task TTL: %s", conversation_id)
