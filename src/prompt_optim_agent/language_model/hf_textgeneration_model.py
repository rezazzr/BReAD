from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_model import BaseLanguageModel


class HFTextGenerationModel(BaseLanguageModel):
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, **kwargs)

        self.device = device or self.get_default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **self.get_tokenizer_kwargs()
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)

    def batch_forward_func(self, batch_prompts):
        model_inputs = self.tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        model_output = self.model.generate(
            **model_inputs,
            do_sample=self.should_sample,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_sequences = model_output.sequences[
            :, model_inputs["input_ids"].size(1) :
        ]
        responses = self.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )

        return [response.strip() for response in responses]

    def generate(self, input):
        model_inputs = self.tokenizer(
            [input], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        model_output = self.model.generate(
            **model_inputs,
            do_sample=self.should_sample,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_sequence = model_output.sequences[
            0, model_inputs["input_ids"].size(1) :
        ]
        response = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)

        if response == "" or response is None:
            return ""
        return response
