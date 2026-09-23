import pytest

from app.core.config import Settings


@pytest.fixture(autouse=True)
def disable_external_credentials_during_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local .env from turning unit tests into paid network calls."""
    monkeypatch.setattr(
        "app.main.settings",
        Settings(_env_file=None, openai_api_key="", tavily_api_key=""),
    )
