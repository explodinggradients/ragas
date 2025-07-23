from .core.common import APP_IDENTIFIER
from .core.app_configuration import AppConfig
from .core.base_logging_service import setup_base_logger

# Configs
app_config = AppConfig.create()

app_log = setup_base_logger(
    APP_IDENTIFIER, app_config.get_log_level()
)
