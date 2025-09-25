"""LLM client wrapper supporting both vLLM and OpenAI."""

import os
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from loguru import logger

from ...utils.config import LLMConfig


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Invoke the LLM with a list of messages."""
        pass
    
    def get_completion(self, prompt: str) -> str:
        """Get completion for a single prompt string."""
        messages = [{"role": "user", "content": prompt}]
        return self.invoke(messages)


class VLLMClient(LLMClient):
    """vLLM client using OpenAI-compatible API."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logger.bind(name="vllm_client")
        
        # Import OpenAI client for vLLM compatibility
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.api_key or "dummy-key",
                base_url=config.vllm_endpoint,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("OpenAI client is required for vLLM support. Install with: pip install openai")
    
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Invoke vLLM with messages."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.vllm_model_name or self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"vLLM request failed: {e}")
            raise


class OpenAIClient(LLMClient):
    """OpenAI client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logger.bind(name="openai_client")
        
        try:
            from openai import OpenAI
            client_kwargs = {
                "api_key": config.api_key,
                "timeout": config.timeout
            }
            
            if config.openai_base_url:
                client_kwargs["base_url"] = config.openai_base_url
            
            self.client = OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("OpenAI client is required. Install with: pip install openai")
    
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Invoke OpenAI with messages."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI request failed: {e}")
            raise


class LangChainVLLMClient(LLMClient):
    """LangChain-based vLLM client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logger.bind(name="langchain_vllm_client")
        
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.config.vllm_model_name or self.config.model,
                openai_api_key=config.api_key or "dummy-key",
                openai_api_base=config.base_url or config.vllm_endpoint,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                request_timeout=config.timeout
            )
        except ImportError:
            raise ImportError("LangChain OpenAI is required. Install with: pip install langchain-openai")
    
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Invoke LangChain vLLM with messages."""
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                else:
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            response = self.llm.invoke(langchain_messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LangChain vLLM request failed: {e}")
            raise


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an appropriate LLM client based on configuration."""
    if config.provider.lower() == "vllm":
        # Try LangChain first, fallback to direct OpenAI client
        try:
            return LangChainVLLMClient(config)
        except ImportError:
            return VLLMClient(config)
    elif config.provider.lower() == "openai":
        return OpenAIClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")


def test_llm_connection(config: LLMConfig) -> bool:
    """Test LLM connection."""
    try:
        client = create_llm_client(config)
        test_messages = [
            {"role": "user", "content": "Hello, this is a test message. Please respond with 'OK'."}
        ]
        response = client.invoke(test_messages)
        logger.info(f"LLM connection test successful. Response: {response[:100]}...")
        return True
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False


