import atexit
import json
import os
import time
from typing import List

import requests
from joblib import Parallel, delayed
from msal import PublicClientApplication, SerializableTokenCache

import wandb


class LLMClientCore:

    _ENDPOINT = "https://fe-26.qas.bing.net/sdf/"

    def __init__(self, set_up_config):
        self.setup_config = set_up_config
        self._SCOPES = [
            "api://{app_id}/access".format(app_id=self.setup_config["app_id"])
        ]
        self._cache = SerializableTokenCache()
        atexit.register(
            lambda: (
                open(".llmapi.bin", "w").write(self._cache.serialize())
                if self._cache.has_state_changed
                else None
            )
        )

        self._app = PublicClientApplication(
            self.setup_config["app_id"],
            authority="https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47",
            token_cache=self._cache,
        )
        if os.path.exists(".llmapi.bin"):
            self._cache.deserialize(open(".llmapi.bin", "r").read())
        print("initiated llm client")

    def send_request(self, model_name, request, _API="completions"):

        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            "X-ScenarioGUID": self.setup_config[
                "scenario_id"
            ],  # special permission to use the API with higher rate limit
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
            "X-ModelType": model_name,
        }

        body = str.encode(json.dumps(request))
        url = LLMClientCore._ENDPOINT + _API
        # print(url)
        response = requests.post(url=url, data=body, headers=headers)
        return response.json()

    def send_stream_request(self, model_name, request, _API="completions"):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
            "X-ModelType": model_name,
        }

        body = str.encode(json.dumps(request))
        response = requests.post(
            LLMClientCore._ENDPOINT + _API, data=body, headers=headers, stream=True
        )
        for line in response.iter_lines():
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
                if text == "[DONE]":
                    break
                else:
                    yield json.loads(text)

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(self._SCOPES, account=chosen)

        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=self._SCOPES)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4)
                )

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        return result["access_token"]


def get_client(setup_config=None):
    assert setup_config is not None, "setup_config is required."
    if "_llm_client" not in globals():
        _llm_client = LLMClientCore(setup_config)
    return _llm_client


class LLMClient:
    def __init__(self, setup_config=None):
        self._llm_client = get_client(setup_config=setup_config)

    def run(self, request_data, model_name, sleep_for_throtlling=5, _API="completions"):
        time.sleep(sleep_for_throtlling)

        while True:
            try:
                start_time = time.time()
                response = self._llm_client.send_request(
                    model_name, request_data, _API=_API
                )
                end_time = time.time()
            except Exception as e:
                print(e)
                continue

            if "choices" not in response:
                print(response)
                if (
                    ("error" in response)
                    and ("code" in response["error"])
                    and (response["error"]["code"] == "content_filter")
                ):
                    print("ignore content filter error")
                    return {
                        "output_msgs": [{"content": ""}],
                        "output_texts": [""],
                        "finish_reasons": [""],
                        "completion_tokens": None,
                        "prompt_tokens": None,
                        "total_tokens": None,
                        "latency": None,
                    }

                time.sleep(50)
                continue
            latency = end_time - start_time
            break

        output_json = {
            "output_msgs": [
                response["choices"][i]["message"]
                for i in range(len(response["choices"]))
                if "message" in response["choices"][i]
            ],
            "output_texts": [
                response["choices"][i]["text"]
                for i in range(len(response["choices"]))
                if "text" in response["choices"][i]
            ],
            "finish_reasons": [
                response["choices"][i]["finish_reason"]
                for i in range(len(response["choices"]))
            ],
            "completion_tokens": response["usage"].get("completion_tokens", None),
            "prompt_tokens": response["usage"].get("prompt_tokens", None),
            "total_tokens": response["usage"].get("total_tokens", None),
            "latency": latency,
        }

        return output_json


MODEL = "dev-gpt-4o-2024-05-13-chat-completions"


class MSFTSubstrate:
    def __init__(
        self,
        model_name: str = MODEL,
        temperature: float = 1.0,
        max_token: int = 4096,
        sleepiness: int = 1,
        **kwargs,
    ):
        self.setup_config = kwargs.get("setup_config", None)
        self.client = LLMClient(setup_config=self.setup_config)
        self.API = "chat/completions"

        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.sleepiness = sleepiness

    def batch_forward_func(self, batch_prompts: List[str]) -> List[str]:
        """
        Generates responses for a batch of prompts.

        Args:
            batch_prompts (List[str]): A list of prompts for which responses need to be generated.

        Returns:
            List[str]: A list of generated responses corresponding to each prompt.

        """
        batch_output = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self.generate)(item) for item in batch_prompts
        )

        generated_text_batch = [item[0] for item in batch_output]
        logging_dict_batch = [item[1] for item in batch_output]
        summed_loging_dict = {}
        for loging_dict in logging_dict_batch:
            for key, value in loging_dict.items():
                summed_loging_dict[key] = summed_loging_dict.get(key, 0) + value

        return generated_text_batch, summed_loging_dict

    def generate(self, prompt: str) -> str:
        """
        Generates text based on the given prompt using the Microsoft Substrate language model.

        Args:
            prompt (str): The input prompt for generating text.
            sleep_for_throtlling (int, optional): The sleep time in seconds for throttling. Defaults to 1.

        Returns: (str): The generated text output.
        """
        request_data: dict = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_token,
            "temperature": self.temperature,
            "stop": [],
        }
        backoff_time: int = 1
        while True:
            try:
                client_response: dict = self.client.run(
                    request_data=request_data,
                    model_name=self.model_name,
                    _API=self.API,
                    sleep_for_throtlling=self.sleepiness,
                )
                loging_dict = {
                    "generated_tokens": client_response["completion_tokens"],
                    "prompt_tokens": client_response["prompt_tokens"],
                    "total_tokens": client_response["total_tokens"],
                    "latency": client_response["latency"],
                }
                wandb.log(loging_dict)
                return (client_response["output_msgs"][0]["content"], loging_dict)
            except Exception as e:
                print(e, f" Sleeping {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
