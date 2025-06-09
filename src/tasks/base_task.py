# define task prompts for various datasets
import json
import os
import random
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""

    train_size: Optional[int] = None
    eval_size: int = 150
    test_size: int = 0
    seed: Optional[int] = None
    base_shuffle: bool = True


class BaseDataset(Dataset):
    """Base dataset class for wrapping data."""

    def __init__(self, dataset: List[Dict[str, Any]]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.dataset[index]


class BaseTask:
    """Base class for various machine learning tasks."""

    def __init__(
        self,
        train_size: Optional[int],
        eval_size: int,
        test_size: int = 0,
        task_name: str = "base_task",
        data_dir: str = "",
        seed: Optional[int] = None,
        post_instruction: bool = False,
        TaskDataset: type = BaseDataset,
        option_num: int = 5,
        **kwargs,
    ):
        """Initialize BaseTask with dataset loading and splitting."""
        self.task_name = task_name
        self.data_dir = data_dir
        self.seed = seed
        self.post_instruction = post_instruction
        self.TaskDataset = TaskDataset
        self.option_num = option_num
        self.answer_format_prompt = "At the end show the answer option bracketed between <answer> and </answer>."

        # Load and process dataset in one pipeline
        self.dataset = self._load_and_split_dataset(
            SplitConfig(train_size, eval_size, test_size, seed)
        )

        # Update sizes and print info
        self._update_and_print_sizes()

    def _load_and_split_dataset(self, config: SplitConfig) -> Dict[str, List[Dict]]:
        """Load and split dataset in one pipeline."""
        origin_dataset = self.load_task_dataset(self.data_dir)
        transformed_dataset = self.transform_format(origin_dataset)
        return self.get_split_task_dataset(transformed_dataset, config)

    def _update_and_print_sizes(self):
        """Update instance sizes and print information."""
        sizes = {split: len(data) for split, data in self.dataset.items()}
        self.train_size, self.eval_size, self.test_size = sizes.values()
        print(
            f"Dataset sizes - train: {self.train_size}, eval: {self.eval_size}, test: {self.test_size}"
        )

    def load_task_dataset(self, data_dir: str) -> List[Dict[str, str]]:
        """Load task dataset from json files."""
        if not data_dir:
            raise ValueError("data_dir cannot be empty")

        dataset = self._load_json_file(data_dir)
        return [
            {"question": ex["question"], "answer": ex["answer"]}
            for ex in dataset["examples"]
        ]

    def transform_format(self, dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Transform dataset format (override in subclasses if needed)."""
        return dataset

    def cal_correct(
        self, preds: List[Any], labels: List[Any], data_type: str = "str", **kwargs
    ) -> List[int]:
        """Compare predictions and labels efficiently."""
        if len(preds) != len(labels):
            raise ValueError(
                f"Length mismatch: preds={len(preds)}, labels={len(labels)}"
            )

        return (
            [1 if pred == label else 0 for pred, label in zip(preds, labels)]
            if data_type == "set"
            else np.equal(preds, labels).astype(int).tolist()  # type: ignore
        )

    def cal_metric(self, preds: List[Any], labels: List[Any], **kwargs) -> float:
        """Calculate accuracy metric."""
        return np.mean(self.cal_correct(preds, labels)).item()

    def cal_metric_from_cal_correct_output(
        self, cal_correct_output: List[int]
    ) -> float:
        """Calculate metric from cal_correct output."""
        return np.mean(cal_correct_output).item()

    def clean_labels(self, labels: List[Any]) -> List[Any]:
        """Clean labels (override in subclasses if needed)."""
        return labels

    def clean_response(self, response: str) -> str:
        """Extract prediction from model response with compact logic."""
        if not response:
            return "N/A: Empty response"

        letters = (
            string.ascii_uppercase[: self.option_num]
            + string.ascii_lowercase[: self.option_num]
        )

        # Extract answer from <answer> tags
        matches = re.findall(r"<answer>([\s\S]*?)<\/answer>", response.lower())
        if not matches:
            return "N/A: Format error"

        # Try to find answer in parentheses first, then standalone
        answer_text = matches[-1]
        for pattern in [rf"\([{letters}]\)", rf"[{letters}]"]:
            match = re.search(pattern, answer_text)
            if match:
                return (
                    match.group(0)[1].upper()
                    if pattern.startswith(r"\(")
                    else match.group(0).upper()
                )

        return "N/A: Format error"

    def batch_clean_responses(self, responses: Union[List[str], Any]) -> List[str]:
        """Extract predictions from batch responses."""
        return [
            self.clean_response(resp)
            for resp in (responses if isinstance(responses, list) else list(responses))
        ]

    def build_forward_prompts_completion(
        self, questions: List[str], cur_prompt: str
    ) -> List[str]:
        """Build prompts by combining questions and current prompt."""
        template = "{}\n{}" if self.post_instruction else "{}\n{}\n{}"
        return [
            (
                template.format(q, cur_prompt)
                if self.post_instruction
                else template.format(cur_prompt, q, self.answer_format_prompt)
            )
            for q in questions
        ]

    def get_split_task_dataset(
        self, origin_dataset: Union[List[Dict], Dict], config: SplitConfig
    ) -> Dict[str, List[Dict]]:
        """Split dataset using configuration object."""
        splitter = self._get_dataset_splitter(origin_dataset)
        train_set, eval_set, test_set = splitter(origin_dataset, config)  # type: ignore
        return {"train": train_set, "eval": eval_set, "test": test_set}

    def _get_dataset_splitter(self, dataset):
        """Get appropriate splitter function based on dataset type."""
        if isinstance(dataset, dict):
            return self._split_dict_dataset
        elif isinstance(dataset, list):
            return self._split_list_dataset
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    def _shuffle_if_needed(
        self, dataset: List[Dict], config: SplitConfig
    ) -> List[Dict]:
        """Shuffle dataset if needed based on configuration."""
        if config.base_shuffle and config.seed is not None:
            print(f"shuffle dataset seed {config.seed}")
            random.seed(config.seed)
            random.shuffle(dataset)
        return dataset

    def _split_dict_dataset(
        self, dataset: Dict, config: SplitConfig
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dictionary-format dataset."""
        train_set = self._shuffle_if_needed(dataset["train"].copy(), config)

        # Find test set from available keys
        test_set = next(
            (
                dataset[key].copy()
                for key in ["test", "validation", "valid"]
                if key in dataset
            ),
            [],
        )

        # Create splits
        eval_set = train_set[-config.eval_size :] if config.eval_size > 0 else []
        train_set = (
            train_set[: config.train_size]
            if config.train_size is not None
            else train_set
        )
        test_set = test_set[: config.test_size] if config.test_size > 0 else []

        return train_set, eval_set, test_set

    def _split_list_dataset(
        self, dataset: List[Dict], config: SplitConfig
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split list-format dataset."""
        # Extract test set first before any shuffling to ensure consistency
        test_set = dataset[: config.test_size] if config.test_size > 0 else []
        remaining = dataset[config.test_size :]

        # Now shuffle only the remaining data (excluding test set)
        remaining = self._shuffle_if_needed(remaining, config)

        # Split the remaining data into train and eval
        train_set = (
            remaining[: config.train_size]
            if config.train_size is not None
            else remaining
        )
        eval_set = remaining[-config.eval_size :] if config.eval_size > 0 else []

        return train_set, eval_set, test_set

    def _load_json_file(self, data_dir: str) -> Dict:
        """Load JSON file with compact error handling."""
        if (
            not data_dir
            or not os.path.exists(data_dir)
            or not data_dir.endswith(".json")
        ):
            raise ValueError(f"Invalid JSON file path: {data_dir}")

        try:
            with open(data_dir, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Error loading JSON file {data_dir}: {e}")

    def build_task_dataset(self, dataset: List[Dict], TaskDataset: type) -> Dataset:
        """Build task dataset using specified Dataset class."""
        return TaskDataset(dataset=dataset)

    def get_dataloader(
        self, split: str, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        """Get DataLoader for specified split with validation."""
        if split not in self.dataset:
            raise ValueError(
                f"Invalid split '{split}'. Available: {list(self.dataset.keys())}"
            )

        dataset = self.build_task_dataset(self.dataset[split], self.TaskDataset)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_dataset_size(self, split: str = "test") -> int:
        """Get size of specified dataset split."""
        if split not in self.dataset:
            raise ValueError(
                f"Invalid split '{split}'. Available: {list(self.dataset.keys())}"
            )
        return len(self.dataset[split])

    def process_gradient_descent_output(self, gradient_descent_output: Any) -> Any:
        """Process gradient descent output (override in subclasses if needed)."""
        return gradient_descent_output
