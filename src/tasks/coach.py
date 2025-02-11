import random
import re
from typing import Dict, List

import numpy as np

from prompt_optim_agent.language_model import get_language_model
from prompt_optim_agent.world_model.prompts import (
    llm_based_metric_evaluation_prompt_tempelate,
)

from .base_task import BaseTask


class CustomTask(BaseTask):
    def __init__(
        self,
        train_size,
        eval_size,
        test_size=None,
        task_name="coach",
        seed=None,
        post_instruction=True,
        data_dir=None,
        **kwargs,
    ):
        self.options = {}
        super().__init__(
            task_name=task_name,
            seed=seed,
            train_size=train_size,
            eval_size=eval_size,
            test_size=test_size,
            post_instruction=post_instruction,
            data_dir=data_dir,
        )
        self.evaluation_model_setting = kwargs.get("evaluation_model_setting", None)
        self.evaluation_model = get_language_model(
            self.evaluation_model_setting["model_type"]
        )(**self.evaluation_model_setting)
        self.metrics_definition = self._load_json_file(
            kwargs.get("metrics_definition_path")
        )
        self.test_metrics_definition = (
            self.metrics_definition
            if kwargs.get("test_metrics_definition_path", None) is None
            else self._load_json_file(kwargs.get("test_metrics_definition_path"))
        )

    def load_task_dataset(self, data_dir: str) -> List[Dict[str, str]]:
        """
        <Task Specific>
        This is a default function for loading task dataset from json files. It can be re-implemented in the task.py files.

        The output dataset can be either a list of question answer pairs or a dict with a default train-test split:
            all examples:
                [{'question':question, 'answer':answer}]
            or
            default split:
                {'train':[{'question':question, 'answer':answer}], 'test':[{'question':question, 'answer':answer}]}
        """
        dataset = self._load_json_file(data_dir)

        return dataset

    def split_dict_dataset(
        self,
        dataset,
        train_size=None,
        eval_size=150,
        test_size=0,
        seed=None,
        base_shuffle=True,
    ):
        train_set = dataset["train"]
        test_set = dataset["test"]
        eval_set = dataset["valid"]

        if base_shuffle and seed is not None:
            print(f"shuffle dataset seed {seed}")
            random.seed(seed)
            random.shuffle(train_set)

        if train_size is not None:
            train_set = train_set[:train_size]
        if test_size is not None:
            test_set = test_set[:test_size]
        if eval_size is not None:
            eval_set = eval_set[:eval_size]
        return train_set, eval_set, test_set

    def clean_response(self, response: str):
        """
        <task specific>
        Extract the prediction from base_model's response,
        so that the output form batch_clean_responses fit
        the input requirement of function "cal_correct"
        """
        clean_pattern = r"<answer>([\s\S]*?)</answer>"
        match = re.findall(clean_pattern, response.lower())
        if len(match) == 0:
            return "N/A: Format error"
        return match[0]

    def cal_correct(
        self,
        preds: List[str],
        questions: List[str],
        prompt: str,
        use_test_metrics: bool = False,
        **kwargs,
    ):
        assert len(prompt) > 0, "Prompt for evaluation is empty!"
        assert len(preds) == len(
            questions
        ), "The number of predictions should be the same as the number of questions."
        results = [
            self.cal_correct_single(
                pred, question, prompt, use_test_metrics=use_test_metrics
            )
            for pred, question in zip(preds, questions)
        ]
        return results

    def cal_metric(
        self,
        preds: List[str],
        questions: List[str],
        prompt: str,
        use_test_metrics: bool = False,
        **kwargs,
    ):
        """
        <task specific>
        Calculate the evaluation metric, e.g. Accuracy, F1 score.
        "question" is the input which includes <prev>, <selected>, and <next> text.
        return a number / tuple of metrics

        This function is for calculating the reward of MCTS.
        """
        matrix_evaluation = np.array(
            self.cal_correct(
                preds=preds,
                questions=questions,
                prompt=prompt,
                use_test_metrics=use_test_metrics**kwargs,
            )
        )
        average_result = matrix_evaluation[matrix_evaluation > -1].mean()
        assert isinstance(
            average_result, float
        ), "The result of the metric should be a float number."
        return average_result

    def cal_metric_from_cal_correct_output(self, cal_correct_output):
        matrix_evaluation = np.array(cal_correct_output)
        average_result = matrix_evaluation[matrix_evaluation > -1].mean()
        assert isinstance(
            average_result, float
        ), "The result of the metric should be a float number."
        return average_result

    def cal_correct_single(
        self,
        pred: str,
        question: str,
        prompt: str,
        use_test_metrics: bool = False,
    ):

        metrics = (
            self.test_metrics_definition
            if use_test_metrics
            else self.metrics_definition
        )
        evaluation_prompts = [
            llm_based_metric_evaluation_prompt_tempelate.format(
                cur_prompt=prompt,
                question=question,
                response=pred,
                metric_name=metric_dict["metric_name"],
                metric_desc=metric_dict["metric_desc"],
                metric_instruction=metric_dict["metric_instruction"],
            )
            for metric_dict in metrics
        ]
        evaluation_responses, _ = self.evaluation_model.batch_forward_func(
            evaluation_prompts
        )
        clean_responses = [
            self.clean_response(response) for response in evaluation_responses
        ]
        int_responses = self.convert_repsonses_to_ints(clean_responses)
        return int_responses

    def convert_repsonses_to_ints(self, responses: List[str]) -> List[int]:
        results = [
            (
                1
                if response.lower() == "yes" or response.lower() == "y"
                else 0 if response.lower() == "no" or response.lower() == "n" else -1
            )
            for response in responses
        ]

        return results
