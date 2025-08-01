import os
from enum import Enum
from typing import Optional

from .common import DEFAULT_BUNDLED_SERVER_PORT
from ..core.base_logging_service import LogLevel


class Environment(Enum):
    DEVELOPMENT = "DEVELOPMENT"
    PRODUCTION = "PRODUCTION"


class AppConfig:
    def __init__(self):
        self._environment = os.getenv("ENVIRONMENT", "PRODUCTION")
        self.port = int(os.getenv("FASTAPI_PORT", str(DEFAULT_BUNDLED_SERVER_PORT)))
        self.project_dir = os.getenv("WEBVIEW_PROJECT_DIR")

    @property
    def is_production(self) -> bool:
        return self._environment == Environment.PRODUCTION.value

    @property
    def is_development(self) -> bool:
        return not self.is_production

    def get_environment(self) -> Environment:
        return Environment.PRODUCTION if self.is_production else Environment.DEVELOPMENT

    def reload_uvicorn(self) -> bool:
        return self.is_development

    def get_port(self) -> int:
        return self.port

    def get_project_dir(self) -> Optional[str]:
        return self.project_dir

    def get_log_level(self) -> LogLevel:
        return LogLevel.INFO if self.is_production else LogLevel.DEBUG

    def get_log_level_value(self) -> str:
        return self.get_log_level().value

    @classmethod
    def create(cls) -> "AppConfig":
        return cls()
