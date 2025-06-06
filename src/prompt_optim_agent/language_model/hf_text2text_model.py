import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base_model import BaseLanguageModel


class HFText2TextModel(BaseLanguageModel):
    def __init__(
        self,
        model_name: str,
        temperature: float,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(model_name, temperature, **kwargs)

        self.device = device
        self.do_sample = True if temperature != 0 else False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding=True, truncate=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)

    def batch_forward_func(self, batch_prompts):
        inputs = self.tokenizer(batch_prompts, return_tensors="pt").to(self.device)
        model_output = self.model.generate(
            **inputs, do_sample=self.do_sample, temperature=self.temperature
        )
        responses = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        return responses

    def generate(self, input):
        inputs = self.tokenizer([input], return_tensors="pt").to(self.device)
        model_output = self.model.generate(
            **inputs, do_sample=self.do_sample, temperature=self.temperature
        )
        responses = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        return responses[0]
