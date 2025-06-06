from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base_model import BaseLanguageModel


class HFText2TextModel(BaseLanguageModel):
    def __init__(
        self,
        model_name: str,
        temperature: float,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, temperature, **kwargs)

        self.device = device or self.get_default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **self.get_tokenizer_kwargs()
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)

    def batch_forward_func(self, batch_prompts):
        inputs = self.tokenizer(batch_prompts, return_tensors="pt").to(self.device)
        model_output = self.model.generate(
            **inputs, do_sample=self.should_sample, temperature=self.temperature
        )
        responses = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        return responses

    def generate(self, input):
        inputs = self.tokenizer([input], return_tensors="pt").to(self.device)
        model_output = self.model.generate(
            **inputs, do_sample=self.should_sample, temperature=self.temperature
        )
        responses = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        return responses[0]
