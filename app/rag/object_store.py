"""MinIO 原件存储；对象名与 Java FileManageService 相同。"""

import asyncio
from io import BytesIO
from urllib.parse import urlsplit

from app.core.config import Settings


def object_name(file_id: str, file_type: str) -> str:
    """输入 fileId/后缀，输出不含用户文件名的稳定对象名，避免路径注入与重名。"""
    return f"file-{file_id.replace('-', '')}.{file_type.lower()}"


class MinioObjectStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        endpoint = settings.minio_endpoint
        if "://" not in endpoint:
            endpoint = f"http://{endpoint}"
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MINIO_ENDPOINT 必须是含 http:// 或 https:// 的 URL")
        self._endpoint = parsed
        self._public_endpoint = endpoint.rstrip("/")
        self._client = None

    def _get_client(self):
        """延迟创建客户端：没配 MinIO 时，健康检查和非文件接口仍能启动。"""
        if self._client is None:
            from minio import Minio

            self._client = Minio(
                self._endpoint.netloc,
                access_key=self._settings.minio_access_key,
                secret_key=self._settings.minio_secret_key,
                secure=self._endpoint.scheme == "https",
            )
        return self._client

    async def put(self, name: str, content: bytes, content_type: str) -> str:
        """输入对象名/字节/MIME，输出可存入 ai_file_info.minio_path 的 URL。

        SDK 是同步的，所以放入线程，不阻塞 FastAPI 事件循环。和 Java 不同，
        新建桶时不设置公开读策略：文件原件不应仅凭 URL 被任何人下载。
        """

        def upload() -> str:
            client = self._get_client()
            bucket = self._settings.minio_bucket_name
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
            client.put_object(
                bucket, name, BytesIO(content), len(content), content_type=content_type
            )
            return f"{self._public_endpoint}/{bucket}/{name}"

        return await asyncio.to_thread(upload)

    async def delete(self, name: str) -> None:
        """输入对象名，输出无；用于删文件时清理 MinIO 原件。"""
        await asyncio.to_thread(
            self._get_client().remove_object, self._settings.minio_bucket_name, name
        )
