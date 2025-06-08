from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TypeVar

State = TypeVar("State")
Action = TypeVar("Action")
Trace = tuple[list[State], list[Action]]


@dataclass
class OptimNode:
    """
    A data class representing a node in an optimization graph.

    Attributes:
        node_id (float): Unique identifier for the node.
        parent_id (int): Identifier of the parent node.
        children_id (List[int]): List of identifiers for the children nodes.
        prompt (str): The prompt or description associated with the node.
        gradient (str): The gradient/feedback or direction for optimization.
    """

    node_id: float
    parent: Optional[int]
    prompt: str
    children_id: List[int] = field(default_factory=list)
    gradient: Optional[str] = None
    kind: str = "optim"

    def __str__(self) -> str:
        """
        Represents the OptimNode instance as a string with its properties in key=value format.

        Returns:
            str: A string representation of the OptimNode instance.
        """
        return f"node_id={self.node_id}\nparent={self.parent}\nchildren_id={self.children_id}\nprompt={self.prompt}\ngradient={self.gradient}\nkind={self.kind}"

    def to_dict(self) -> dict:
        """
        Converts the OptimNode instance into a dictionary.

        Returns:
            dict: A dictionary representation of the OptimNode instance with its properties as key-value pairs.
        """
        return {
            "node_id": self.node_id,
            "parent": self.parent,
            "prompt": self.prompt,
            "gradient": self.gradient,
            "kind": self.kind,
        }


class SearchAlgo(ABC):
    def __init__(
        self,
        task,
        world_model,
        action_agent,
        logger=None,
        seed=0,
        print_log=True,
        test_every_step=True,
        depth_limit=None,
    ) -> None:
        self.task = task
        self.world_model = world_model
        self.action_agent = action_agent
        self.states = []
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.seed = seed
        self.test_every_step = test_every_step
        self.depth_limit = depth_limit

    @abstractmethod
    def search(self):
        pass

    def get_states(self):
        return self.states

    def process_all_correct_batch(self):
        if self.logger is None:
            return
        self.logger.info("\n-----------------------------------------------------")
        self.logger.info("all correct: skip updating cur_prompt")
        self.logger.info("\n-----------------------------------------------------\n")
