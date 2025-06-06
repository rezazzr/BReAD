import time

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from .base_model import BaseLanguageModel

CHAT_COMPLETION_MODELS = [
    "gpt-4o",
    "gpt-4-turbo-preview",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo-16k",
]


class OpenAIModel(BaseLanguageModel):
    def __init__(self, model_name: str, api_key: str, temperature: float, **kwargs):
        super().__init__(model_name, temperature, **kwargs)

        if api_key is None:
            raise ValueError(f"api_key error: {api_key}")
        try:
            self.model = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Init openai client error: \n{e}")
            raise RuntimeError("Failed to initialize OpenAI client") from e

        if model_name in CHAT_COMPLETION_MODELS:
            self.batch_forward_func = self.batch_forward_chatcompletion
            self.generate = self.gpt_chat_completion
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def batch_forward_chatcompletion(self, batch_prompts):
        """
        Input a batch of prompts to openai chat API and retrieve the answers.
        """
        responses = []
        for prompt in batch_prompts:
            response = self.gpt_chat_completion(input=prompt)
            responses.append(response)
        return responses

    def gpt_chat_completion(self, input):

        messages = [
            ChatCompletionUserMessageParam(role="user", content=input),
        ]
        backoff_time = 1
        while True:
            try:
                return (
                    self.model.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=self.temperature,
                    )
                    .choices[0]
                    .message.content.strip()  # type: ignore
                )
            except Exception as e:
                print(e, f" Sleeping {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
                time.sleep(backoff_time)
                backoff_time *= 1.5
                backoff_time *= 1.5
