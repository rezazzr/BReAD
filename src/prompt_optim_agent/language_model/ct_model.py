from typing import Optional

import ctranslate2
import transformers

from .base_model import BaseLanguageModel


class CTranslateModel(BaseLanguageModel):
    def __init__(
        self,
        model_name: str,
        # huggingface model name, e.g. "mistralai/Mistral-7B-v0.1"
        model_path: str,
        # your downloaded ct model path, e.g. "./workspace/download_models/Mistral-7B-Instruct-v0.2_int8_float16"
        temperature: float = 0,
        max_length: int = 512,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, temperature, **kwargs)

        self.device = device or self.get_default_device()
        self.max_length = max_length
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, **self.get_tokenizer_kwargs()
        )
        self.model = ctranslate2.Generator(model_path, device=device)

    def batch_forward_func(self, batch_prompts):
        return self.default_batch_forward_func(batch_prompts)

    def generate(self, input):
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(input))
        results = self.model.generate_batch(
            [tokens],
            sampling_temperature=self.temperature,
            max_length=self.max_length,
            include_prompt_in_result=False,
        )
        output = self.tokenizer.decode(results[0].sequences_ids[0])
        return output
