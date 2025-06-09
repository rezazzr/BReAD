from typing import Dict, List, Optional, Tuple, Union

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
        max_tokens: int = 512,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, **kwargs)

        self.device = device or self.get_default_device()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, **self.get_tokenizer_kwargs()
        )
        self.model = ctranslate2.Generator(model_path, device=device)

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
            tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(input))
            results = self.model.generate_batch(
                [tokens],
                sampling_temperature=self.temperature,
                max_length=self.max_tokens,
                include_prompt_in_result=False,
            )
            output = self.tokenizer.decode(results[0].sequences_ids[0])
            return output, tokens, results[0].sequences_ids[0]

        (output, input_tokens, output_tokens), latency = self.timed_call(call_model)

        logging_dict = {
            "generated_tokens": len(output_tokens),
            "prompt_tokens": len(input_tokens),
            "total_tokens": len(input_tokens) + len(output_tokens),
            "latency": latency,
        }
        return output, logging_dict
