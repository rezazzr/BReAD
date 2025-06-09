from typing import Dict, List, Optional, Tuple, Union

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
            response = self.tokenizer.decode(
                generated_sequence, skip_special_tokens=True
            )

            return (
                response,
                model_inputs["input_ids"].size(1),
                generated_sequence.size(0),
            )

        (response, input_length, generated_length), latency = self.timed_call(
            call_model
        )

        logging_dict = {
            "generated_tokens": generated_length,
            "prompt_tokens": input_length,
            "total_tokens": input_length + generated_length,
            "latency": latency,
        }
        return response, logging_dict
