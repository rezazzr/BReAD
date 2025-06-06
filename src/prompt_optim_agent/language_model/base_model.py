from abc import ABC, abstractmethod
from typing import List


class BaseLanguageModel(ABC):
    """Base class for all language models."""

    def __init__(self, model_name: str, temperature: float, **kwargs):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def batch_forward_func(self, batch_prompts: List[str]) -> List[str]:
        """
        Process a batch of prompts and return responses.

        Args:
            batch_prompts: List of input prompts

        Returns:
            List of generated responses
        """
        pass

    @abstractmethod
    def generate(self, input: str) -> str:
        """
        Generate a response for a single input prompt.

        Args:
            input: Input prompt string

        Returns:
            Generated response string
        """
        pass
