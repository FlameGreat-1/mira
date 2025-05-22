import json
import threading
import tomllib
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", PROJECT_ROOT / "workspace"))


class LLMSettings(BaseModel):
    model: str = Field(...)
    base_url: str = Field(...)
    api_key: str = Field(...)
    max_tokens: int = Field(4096)
    max_input_tokens: Optional[int] = Field(None)
    temperature: float = Field(1.0)
    api_type: str = Field("openai")
    api_version: Optional[str] = Field(None)
    endpoint_id: Optional[str] = Field(None)
    timeout: int = Field(120)
    retry_count: int = Field(3)
    streaming_supported: bool = Field(True)


class ProxySettings(BaseModel):
    server: str = Field(None)
    username: Optional[str] = Field(None)
    password: Optional[str] = Field(None)


class SearchSettings(BaseModel):
    engine: str = Field(default="Google")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"]
    )
    retry_delay: int = Field(default=60)
    max_retries: int = Field(default=3)
    lang: str = Field(default="en")
    country: str = Field(default="us")


class BrowserSettings(BaseModel):
    headless: bool = Field(False)
    disable_security: bool = Field(True)
    extra_chromium_args: List[str] = Field(default_factory=list)
    chrome_instance_path: Optional[str] = Field(None)
    wss_url: Optional[str] = Field(None)
    cdp_url: Optional[str] = Field(None)
    proxy: Optional[ProxySettings] = Field(None)
    max_content_length: int = Field(2000)


class SandboxSettings(BaseModel):
    use_sandbox: bool = Field(False)
    image: str = Field("python:3.12-slim")
    work_dir: str = Field("/workspace")
    memory_limit: str = Field("512m")
    cpu_limit: float = Field(1.0)
    timeout: int = Field(300)
    network_enabled: bool = Field(False)


class MCPServerConfig(BaseModel):
    type: str = Field(...)
    url: Optional[str] = Field(None)
    command: Optional[str] = Field(None)
    args: List[str] = Field(default_factory=list)


class MCPSettings(BaseModel):
    server_reference: str = Field("app.mcp.server")
    servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    @classmethod
    def load_server_config(cls) -> Dict[str, MCPServerConfig]:
        config_path = PROJECT_ROOT / "config" / "mcp.json"

        try:
            config_file = config_path if config_path.exists() else None
            if not config_file:
                return {}

            with config_file.open() as f:
                data = json.load(f)
                servers = {}

                for server_id, server_config in data.get("mcpServers", {}).items():
                    servers[server_id] = MCPServerConfig(
                        type=server_config["type"],
                        url=server_config.get("url"),
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                    )
                return servers
        except Exception as e:
            raise ValueError(f"Failed to load MCP server config: {e}")


class AuthSettings(BaseModel):
    # Security settings
    SECRET_KEY: str = Field(default_factory=lambda: os.environ.get("SECRET_KEY", os.urandom(24).hex()))
    ADMIN_USERNAME: str = Field(default=os.environ.get("ADMIN_USERNAME", "admin"))
    ADMIN_PASSWORD: str = Field(default=os.environ.get("ADMIN_PASSWORD", ""))
    
    # JWT Authentication settings
    JWT_ALGORITHM: str = Field(default=os.environ.get("JWT_ALGORITHM", "HS256"))
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440")))
    ALLOW_ADMIN_REGISTRATION: bool = Field(default=os.environ.get("ALLOW_ADMIN_REGISTRATION", "False").lower() == "true")
    
    # Email settings for password reset
    SMTP_HOST: Optional[str] = Field(default=os.environ.get("SMTP_HOST"))
    SMTP_PORT: Optional[int] = Field(default=int(os.environ.get("SMTP_PORT", "587")) if os.environ.get("SMTP_PORT") else None)
    SMTP_USERNAME: Optional[str] = Field(default=os.environ.get("SMTP_USERNAME"))
    SMTP_PASSWORD: Optional[str] = Field(default=os.environ.get("SMTP_PASSWORD"))
    SMTP_USE_TLS: bool = Field(default=os.environ.get("SMTP_USE_TLS", "True").lower() == "true")
    EMAIL_FROM: str = Field(default=os.environ.get("EMAIL_FROM", "noreply@openagentframework.com"))
    FRONTEND_URL: str = Field(default=os.environ.get("FRONTEND_URL", "http://localhost:3000"))
    
    # Database settings
    DATABASE_URL: str = Field(default=os.environ.get("DATABASE_URL", ""))
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v and os.environ.get("ENV") == "production":
            raise ValueError("DATABASE_URL must be set in production environment")
        return v
    
    @validator("ADMIN_PASSWORD")
    def validate_admin_password(cls, v, values):
        if os.environ.get("ENV") == "production" and not v:
            raise ValueError("ADMIN_PASSWORD must be set in production environment")
        return v


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    sandbox: Optional[SandboxSettings] = Field(None)
    browser_config: Optional[BrowserSettings] = Field(None)
    search_config: Optional[SearchSettings] = Field(None)
    mcp_config: Optional[MCPSettings] = Field(None)
    auth: AuthSettings = Field(default_factory=AuthSettings)

    class Config:
        arbitrary_types_allowed = True


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", "openai"),
            "api_version": base_llm.get("api_version"),
            "endpoint_id": base_llm.get("endpoint_id"),
            "timeout": base_llm.get("timeout", 120),
            "retry_count": base_llm.get("retry_count", 3),
            "streaming_supported": base_llm.get("streaming_supported", True),
        }

        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)
        sandbox_config = raw_config.get("sandbox", {})
        if sandbox_config:
            sandbox_settings = SandboxSettings(**sandbox_config)
        else:
            sandbox_settings = SandboxSettings()

        mcp_config = raw_config.get("mcp", {})
        mcp_settings = None
        if mcp_config:
            mcp_config["servers"] = MCPSettings.load_server_config()
            mcp_settings = MCPSettings(**mcp_config)
        else:
            mcp_settings = MCPSettings(servers=MCPSettings.load_server_config())

        # Add auth settings
        auth_settings = AuthSettings()

        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
            "sandbox": sandbox_settings,
            "browser_config": browser_settings,
            "search_config": search_settings,
            "mcp_config": mcp_settings,
            "auth": auth_settings,
        }

        self._config = AppConfig(**config_dict)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm

    @property
    def sandbox(self) -> SandboxSettings:
        return self._config.sandbox

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
        return self._config.search_config

    @property
    def mcp_config(self) -> MCPSettings:
        return self._config.mcp_config
    
    @property
    def auth(self) -> AuthSettings:
        return self._config.auth

    @property
    def workspace_root(self) -> Path:
        return WORKSPACE_ROOT

    @property
    def root_path(self) -> Path:
        return PROJECT_ROOT
    
    # Auth-specific properties for compatibility
    @property
    def SECRET_KEY(self) -> str:
        return self.auth.SECRET_KEY
    
    @property
    def DATABASE_URL(self) -> str:
        return self.auth.DATABASE_URL
    
    @property
    def SMTP_HOST(self) -> Optional[str]:
        return self.auth.SMTP_HOST
    
    @property
    def SMTP_PORT(self) -> Optional[int]:
        return self.auth.SMTP_PORT
    
    @property
    def SMTP_USERNAME(self) -> Optional[str]:
        return self.auth.SMTP_USERNAME
    
    @property
    def SMTP_PASSWORD(self) -> Optional[str]:
        return self.auth.SMTP_PASSWORD
    
    @property
    def EMAIL_FROM(self) -> str:
        return self.auth.EMAIL_FROM
    
    @property
    def FRONTEND_URL(self) -> str:
        return self.auth.FRONTEND_URL
    
    @property
    def DEBUG(self) -> bool:
        return os.environ.get("DEBUG", "False").lower() == "true"


config = Config()
