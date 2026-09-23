import asyncio

import pytest

from app.runtime.task_manager import AgentTaskManager


class FakePubSub:
    def __init__(self, redis: "FakeRedis") -> None:
        self.redis = redis
        self.channel: str | None = None
        self.messages: asyncio.Queue[dict] = asyncio.Queue()

    async def subscribe(self, channel: str) -> None:
        self.channel = channel
        self.redis.subscribers.append(self)

    async def unsubscribe(self, channel: str) -> None:
        assert channel == self.channel
        self.redis.subscribers.remove(self)

    async def aclose(self) -> None:
        pass

    async def listen(self):
        while True:
            yield await self.messages.get()


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.subscribers: list[FakePubSub] = []
        self.published: list[tuple[str, str]] = []

    async def ping(self) -> bool:
        return True

    def pubsub(self) -> FakePubSub:
        return FakePubSub(self)

    async def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
        assert nx
        if key in self.values:
            return False
        self.values[key] = value
        self.ttls[key] = ex
        return True

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def exists(self, key: str) -> int:
        return int(key in self.values)

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    async def expire(self, key: str, ttl: int) -> bool:
        self.ttls[key] = ttl
        return True

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        for subscriber in self.subscribers:
            if subscriber.channel == channel:
                subscriber.messages.put_nowait({"type": "message", "data": message})
        return len(self.subscribers)

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_task_registration_blocks_same_conversation_and_releases_redis_key() -> None:
    redis = FakeRedis()
    manager = AgentTaskManager(redis)
    await manager.start()
    task = asyncio.create_task(asyncio.Event().wait())
    try:
        assert await manager.register_task("conversation-1", task, "web react")
        assert not await manager.register_task("conversation-1", task, "web react")
        assert redis.values["agent:task:conversation-1"] == manager.instance_id
        assert redis.ttls["agent:task:conversation-1"] == 1800
        assert await manager.has_running_task("conversation-1")
        await manager.refresh_task_ttls()
        assert redis.ttls["agent:task:conversation-1"] == 1800
        await manager.unregister_task("conversation-1", task)
        assert "agent:task:conversation-1" not in redis.values
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
async def test_remote_stop_uses_java_pubsub_channel_and_cancels_owner_task() -> None:
    redis = FakeRedis()
    owner = AgentTaskManager(redis)
    other = AgentTaskManager(redis)
    await owner.start()
    await other.start()
    task = asyncio.create_task(asyncio.Event().wait())
    try:
        assert await owner.register_task("conversation-2", task, "web react")
        assert not await other.register_task("conversation-2", task, "web react")
        assert await other.stop_task("conversation-2")
        assert redis.published == [("agent:stop", "conversation-2")]
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        await owner.unregister_task("conversation-2", task)
        assert not await other.has_running_task("conversation-2")
        assert not await other.stop_task("conversation-2")
    finally:
        await owner.close()
        await other.close()


@pytest.mark.asyncio
async def test_local_stop_does_not_publish_and_only_owner_removes_key() -> None:
    redis = FakeRedis()
    manager = AgentTaskManager(redis)
    await manager.start()
    task = asyncio.create_task(asyncio.Event().wait())
    try:
        assert await manager.register_task("conversation-3", task, "web react")
        assert await manager.stop_task("conversation-3")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert redis.published == []
        redis.values["agent:task:conversation-3"] = "another-instance"
        await manager.unregister_task("conversation-3", task)
        assert redis.values["agent:task:conversation-3"] == "another-instance"
    finally:
        await manager.close()
