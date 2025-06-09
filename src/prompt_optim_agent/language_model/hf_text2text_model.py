from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base_model import BaseLanguageModel


class HFText2TextModel(BaseLanguageModel):
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, **kwargs)

        self.device = device or self.get_default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **self.get_tokenizer_kwargs()
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)

    def batch_forward_func(
        self, batch_prompts
    ) -> Tuple[List[str], Dict[str, Union[int, float]]]:
        def process_batch():
            results = []
            for prompt in batch_prompts:
                result = self.generate(prompt)
                results.append(result)
            return results

        results, batch_latency = self.timed_call(process_batch)
        responses = [result_tuple[0] for result_tuple in results]
        # Aggregate logging info
        total_prompt_tokens = sum(
            result_tuple[1]["prompt_tokens"] for result_tuple in results
        )
        total_generated_tokens = sum(
            result_tuple[1]["generated_tokens"] for result_tuple in results
        )
        total_tokens = sum(result_tuple[1]["total_tokens"] for result_tuple in results)
        logging_dict = {
            "generated_tokens": total_generated_tokens,
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_tokens,
            "latency": batch_latency,
        }
        return responses, logging_dict

    def generate(self, input) -> Tuple[str, Dict[str, Union[int, float]]]:
        def call_model():
            inputs = self.tokenizer([input], return_tensors="pt").to(self.device)
            model_output = self.model.generate(
                **inputs, do_sample=self.should_sample, temperature=self.temperature
            )
            responses = self.tokenizer.batch_decode(
                model_output, skip_special_tokens=True
            )
            return responses[0], inputs["input_ids"].shape[1], model_output.shape[1]

        (response, input_length, output_length), latency = self.timed_call(call_model)
        generated_tokens = output_length - input_length

        logging_dict = {
            "generated_tokens": generated_tokens,
            "prompt_tokens": input_length,
            "total_tokens": output_length,
            "latency": latency,
        }
        return response, logging_dict
