from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import torch


class BaseLanguageModel(ABC):
    """Base class for all language models."""

    def __init__(self, model_name: str, temperature: float, max_tokens: int, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def get_default_device() -> str:
        """Get default device based on CUDA availability."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def should_sample(self) -> bool:
        """Determine if sampling should be used based on temperature."""
        return self.temperature != 0

    def get_tokenizer_kwargs(self) -> dict:
        """Get common tokenizer initialization kwargs."""
        return {"trust_remote_code": True, "truncate": True, "padding": True}

    def default_batch_forward_func(self, batch_prompts: List[str]) -> List[str]:
        """Default implementation for batch processing using single generate calls."""
        responses = []
        for prompt in batch_prompts:
            responses.append(self.generate(input=prompt))
        return responses

    @abstractmethod
    def batch_forward_func(
        self, batch_prompts: List[str]
    ) -> Tuple[List[str], Dict[str, Union[int, float]]]:
        """
        Process a batch of prompts and return responses.

        Args:
            batch_prompts: List of input prompts

        Returns:
            Tuple containing a list of generated responses and a single
            dictionary of additional information describing the batch, such as token usage.
        """
        pass

    @abstractmethod
    def generate(self, input: str) -> Tuple[str, Dict[str, Union[int, float]]]:
        """
        Generate a response for a single input prompt.

        Args:
            input: Input prompt string

        Returns:
            Tuple containing the generated response and a dictionary of additional information such as token usage.
        """
        pass
