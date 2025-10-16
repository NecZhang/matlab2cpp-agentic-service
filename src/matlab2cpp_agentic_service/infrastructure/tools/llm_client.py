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


def create_selective_llm_clients(base_config: LLMConfig, selective_config=None) -> Dict[str, LLMClient]:
    """
    Create selective LLM clients for different agents.
    
    Args:
        base_config: Base LLM configuration (fallback)
        selective_config: Selective LLM configuration from environment variables
        
    Returns:
        Dictionary mapping agent names to their specific LLM clients
    """
    clients = {}
    
    # Use selective config if available, otherwise fall back to base config
    if selective_config is None:
        # Fallback to base config for both models
        reasoning_config = LLMConfig(
            provider=base_config.provider,
            model=base_config.model,
            api_key=base_config.api_key,
            temperature=base_config.temperature,
            max_tokens=base_config.max_tokens,
            timeout=base_config.timeout,
            vllm_endpoint=base_config.vllm_endpoint,
            vllm_model_name=base_config.vllm_model_name,
            openai_base_url=base_config.openai_base_url,
            base_url=base_config.base_url
        )
        
        cpp_config = LLMConfig(
            provider=base_config.provider,
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            api_key=base_config.api_key,
            temperature=0.1,
            max_tokens=base_config.max_tokens,
            timeout=base_config.timeout,
            vllm_endpoint="http://192.168.6.19:8011/v1",
            vllm_model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            openai_base_url=base_config.openai_base_url,
            base_url="http://192.168.6.19:8011/v1"
        )
    else:
        # Use selective configuration from environment variables
        reasoning_config = LLMConfig(
            provider=selective_config.general_provider,
            model=selective_config.general_model,
            api_key=selective_config.general_api_key,
            temperature=selective_config.general_temperature,
            max_tokens=selective_config.general_max_tokens,
            timeout=selective_config.general_timeout,
            vllm_endpoint=selective_config.general_vllm_endpoint,
            vllm_model_name=selective_config.general_vllm_model_name,
            openai_base_url=base_config.openai_base_url,
            base_url=selective_config.general_base_url
        )
        
        cpp_config = LLMConfig(
            provider=selective_config.coding_provider,
            model=selective_config.coding_model,
            api_key=selective_config.coding_api_key,
            temperature=selective_config.coding_temperature,
            max_tokens=selective_config.coding_max_tokens,
            timeout=selective_config.coding_timeout,
            vllm_endpoint=selective_config.coding_vllm_endpoint,
            vllm_model_name=selective_config.coding_vllm_model_name,
            openai_base_url=base_config.openai_base_url,
            base_url=selective_config.coding_base_url
        )
    
    # Create clients for different agent types
    try:
        # Reasoning model for most agents (analysis, planning, quality assessment, etc.)
        logger.info(f"DEBUG: Creating reasoning client with config - model: {reasoning_config.model}, endpoint: {reasoning_config.vllm_endpoint}")
        clients["reasoning"] = create_llm_client(reasoning_config)
        logger.info("Created reasoning model client for analysis, planning, and assessment agents")
        
        # Direct model for C++ generation
        logger.info(f"DEBUG: Creating coding client with config - model: {cpp_config.model}, endpoint: {cpp_config.vllm_endpoint}")
        clients["cpp_generation"] = create_llm_client(cpp_config)
        logger.info("Created direct model client for C++ generation agent")
        
    except Exception as e:
        logger.error(f"Failed to create selective LLM clients: {e}")
        # Fallback to single client
        fallback_client = create_llm_client(base_config)
        clients["reasoning"] = fallback_client
        clients["cpp_generation"] = fallback_client
        logger.warning("Using fallback single client for all agents")
    
    return clients


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


