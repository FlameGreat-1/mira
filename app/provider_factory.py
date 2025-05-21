from typing import Dict, Any, Type

from openai import AsyncOpenAI, AsyncAzureOpenAI
from app.bedrock import BedrockClient
from app.runpod_client import AsyncRunPodClient

class ProviderFactory:
    _providers: Dict[str, Type] = {
        "openai": AsyncOpenAI,
        "azure": AsyncAzureOpenAI,
        "aws": BedrockClient,
        "runpod": AsyncRunPodClient,
    }
    
    @classmethod
    def register_provider(cls, provider_type: str, provider_class: Type) -> None:
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def create(cls, provider_type: str, **config) -> Any:
        if provider_type not in cls._providers:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        
        if provider_type == "openai":
            return provider_class(
                api_key=config.get("api_key"),
                base_url=config.get("base_url")
            )
        elif provider_type == "azure":
            return provider_class(
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                api_version=config.get("api_version")
            )
        elif provider_type == "aws":
            return provider_class()
        elif provider_type == "runpod":
            return provider_class(
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                timeout=config.get("timeout", 120),
                retry_count=config.get("retry_count", 3)
            )
        
        return provider_class(**config)
