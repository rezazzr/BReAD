# The MCTS algorithm code is adapted from Reasoning with Language Model is Planning with World Model
# https://github.com/Ber666/llm-reasoners

import itertools
import json
import logging
import os
from copy import deepcopy
from typing import Generic, List, Optional, Tuple

import numpy as np
import wandb

from src.prompt_optim_agent.utils import create_logger
from src.prompt_optim_agent.world_model.world_model import WorldModel

from .base_algo import Action, OptimNode, SearchAlgo, State


class MCTSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        prompt: str,
        action: Optional[Action],
        parent: "Optional[MCTSNode]" = None,
    ):
        """
        A node in the MCTS search tree

        :param prompt: the current state
        :param action: the action of the last optimization step,
            i.e., the state transition prompt from parent node to current node
        :param parent: the parent node, None if root of the tree
        """
        self.id = next(MCTSNode.id_iter)

        self.prompt = prompt
        self.action = action
        self.parent = parent
        self.is_terminal = False

        self.children: list[MCTSNode] = []
        self.cum_rewards: list[float] = []
        self.reward = 0.0
        self.test_metric = -1.0
        self.uct = 0.0

        self.visited = 0
        self.expand_times = 0

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def calc_q(self, x):
        return np.mean(x)

    def cal_reward(self):
        return self.reward

    @property
    def Q(self) -> float:
        if len(self.cum_rewards) == 0:
            return self.reward
        else:
            return self.calc_q(self.cum_rewards)

    def to_dict(self):
        return {
            "id": self.id,
            "depth": self.depth,
            "parent": -1 if self.parent is None else self.parent.id,
            "visited": self.visited,
            "expand_times": self.expand_times,
            "q": self.Q,
            "uct": self.uct,
            "prompt": self.prompt,
            "reward": self.reward,
            "test_metric": self.test_metric,
        }


class MCTS(SearchAlgo, Generic[State, Action]):

    def __init__(
        self,
        task,
        world_model: WorldModel,
        # mcts arguments
        expand_width=3,
        w_exp: float = 2.5,
        depth_limit: int = 8,
        min_depth: int = 2,
        iteration_num: int = 12,
        # log
        log=True,
        logger: Optional[logging.Logger] = None,
        log_dir=None,
        **kwargs,
    ) -> None:
        """
        MCTS search algorithm

        :param task: the specific task
        :param world_model: the MCTS world model for state transition
        :param expand_width: number of batches to be sampled
        :param w_exp: the weight of mcts exploration
        :param depth_limit: the max depth of a single MCTS path
        :param iteration_num: number of MCTS iterations
        :param logger: logger
        :param log_dir: logger directory to save the results
        """

        self.task = task
        self.world_model = world_model

        self.expand_width = expand_width
        self.depth_limit = depth_limit
        self.w_exp = w_exp
        self.iteration_num = iteration_num
        self.min_depth = (
            min_depth  # Apply early stop only when depth is larger than min_depth
        )

        self.mcts_threshold = 0.0  # The highest reward node globally
        self.min_threshold = 0.0  # The root node's reward as a min threshold

        # output
        self.log_dir = log_dir if log_dir is not None else os.getcwd()
        self.logger = (
            logger
            if logger is not None
            else create_logger(self.log_dir, "mcts", log_mode="train")
        )

        self.k = 1  # top-k reward nodes
        self.trace_in_each_iter: list[list[MCTSNode]] = []
        self.root: Optional[MCTSNode] = None
        self.nodes: list[MCTSNode] = []
        self.optim_nodes = []
        self.optim_nodes_ids_only = []
        self.base_optim_node_id = -1
        self.num_gradient_accumulation = kwargs.get("num_gradient_accumulation", 1)
        self.log = log

        self.log_vars()

    def get_optim_id(self) -> int:
        self.base_optim_node_id -= 1
        return self.base_optim_node_id

    def simulate_choice(self, x):
        return np.argmax(x)

    def increase_threshold(self, threshold):
        if threshold > self.mcts_threshold:
            self.mcts_threshold = threshold

    def cal_cum_reward(self, rewards):
        return np.sum(rewards)

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.depth >= self.depth_limit

    def early_stop(self, node: MCTSNode):
        return node.reward > self.mcts_threshold and node.depth > self.min_depth

    def _is_terminal_with_min_threshold(self, node: MCTSNode):
        if node.parent is None:
            min_threshold = self.min_threshold
        else:
            min_threshold = (self.min_threshold + node.parent.reward) / 2
        return node.reward < min_threshold and node.depth > self.min_depth

    def is_terminal_node(self, node: MCTSNode):
        return (
            self._is_terminal_with_depth_limit(node)
            or self._is_terminal_with_min_threshold(node)
            or node.is_terminal
        )

    def _uct(self, node: MCTSNode) -> float:
        if node.parent is None:
            N_parent = 0
        else:
            N_parent = len(node.parent.cum_rewards)
        return node.Q + self.w_exp * np.sqrt(
            np.log(N_parent + 1) / max(1, len(node.cum_rewards))
        )

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        return max(node.children or [], key=self._uct)

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        """
        Selection:
            From root node, keep selecting child node based on UCT
        """

        path = []
        while True:
            path.append(node)
            node.visited += 1
            if len(node.children) == 0 or self.is_terminal_node(node):
                return path

            node = self._uct_select(node)
            if self.log:
                self.logger.info(
                    f"Select node {node.id}: depth {node.depth}, \
                                 reward: {node.reward:.4f} utc: {self._uct(node=node)}"
                )

    def _expand(self, node: MCTSNode):
        """
        Expansion:
            Sample batches of data and perform state transition on the given node.
            Generate new child nodes and calculate their temporary reward.
        """
        if self.log:
            self.logger.info("Expanding:")
        if self.is_terminal_node(node):
            node.is_terminal = True
            return

        if self.log:
            self.logger.info(
                f"Expanding: node: {node.id}, depth {node.depth}, reward: {node.reward:.4f}"
            )

        i = 0
        node.expand_times += 1
        if node.id not in self.optim_nodes_ids_only:
            self.optim_nodes.append(
                OptimNode(
                    node_id=node.id,
                    parent=node.parent.id if node.parent is not None else None,
                    children_id=[],
                    prompt=node.prompt,
                    gradient=None,
                    kind="node",
                )
            )
            self.optim_nodes_ids_only.append(node.id)
        while i < self.expand_width:
            batch = self.world_model.get_train_batch()  # sample batch data
            children, gradient_descent_output = self.world_model.step(node, batch)
            optim_node_id = self.get_optim_id()
            self.optim_nodes.append(
                OptimNode(
                    node_id=optim_node_id,
                    parent=node.id,
                    children_id=[child.id for child in children],
                    prompt=gradient_descent_output["gradient_prompt"],
                    gradient=gradient_descent_output["gradient"],
                )
            )
            self.optim_nodes.extend(
                [
                    OptimNode(
                        node_id=child.id,
                        parent=optim_node_id,
                        children_id=[],
                        prompt=child.prompt,
                        gradient=None,
                        kind="node",
                    )
                    for child in children
                ]
            )
            self.optim_nodes_ids_only.extend(
                [child.id for child in children] + [optim_node_id]
            )

            # optim step: sample new child nodes using one batch

            i += 1
            for (
                child_node
            ) in (
                children
            ):  # There could be multiple children in one optim step (num_new_prompts>1)
                self.world_model.evaluate_child_node(node=child_node)
                child_node.reward = child_node.cal_reward()
                child_node.is_terminal = self.is_terminal_node(child_node)

            self.nodes.extend(children)
            node.children.extend(children)

        if self.log:
            for child in node.children:
                self.logger.info(
                    f"child_node {child.id} (reward:{child.reward:.4f}, reward: {child.reward:.4f})"
                )

    def _simulate(self, path: list[MCTSNode]):
        """
        Simulation: simulate the last node in the selected path, stop if reaching terminal or early stop.
        """

        if self.log:
            self.logger.info("Simulating:")
        node = path[-1]

        while True:
            if self.early_stop(node):
                node.is_terminal = self.is_terminal_node(node)
                self.increase_threshold(node.reward)
                if self.log:
                    self.logger.info(
                        f"Early Stop: node {node.id}, reward: {node.reward}. \
                    MCTS threshold increases to {self.mcts_threshold}. Stop simulating.\n"
                    )
                return

            self.increase_threshold(node.reward)

            if self.is_terminal_node(node):
                return

            if len(node.children) == 0:
                self._expand(node)

            rewards = [child.reward for child in node.children]
            if len(rewards) != 0:
                node = node.children[self.simulate_choice(rewards)]
            else:
                node.is_terminal = True

            node.visited += 1
            path.append(node)

    def _back_propagate(self, path: list[MCTSNode]) -> List[float]:
        """
        Back Propagation: Update the cumulated rewards of each node in the path.
        """
        if self.log:
            self.logger.info("Back propagating:")

        cum_rewards = []
        running_sum = 0.0

        # Traverse from leaf to root
        for node in reversed(path):
            running_sum += node.reward
            cum_rewards.append(running_sum)
            node.cum_rewards.append(running_sum)
            if self.log:
                self.logger.info(
                    f"node {node.id}: depth {node.depth}, new cum_reward: {node.cum_rewards[-1]:.4f}"
                )

        # Reverse to match the original path order (root to leaf)
        cum_rewards.reverse()
        return cum_rewards

    def iterate(self, node: MCTSNode) -> Tuple[list[MCTSNode], List[float]]:
        """
        MCTS iteration: Selection, Expansion, Simulation, Back-Propagation
        """
        path = self._select(node)
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        cum_rewards = self._back_propagate(path)

        return path, cum_rewards

    def search(self, init_state: str):

        self.root = self.world_model.build_root(init_state)
        self.root.reward = self.root.cal_reward()
        self.nodes.append(self.root)

        if self.min_threshold == 0:  # TODO: Experiment with this condition
            self.min_threshold = self.root.reward
            self.increase_threshold(self.root.reward)

        self.trace_in_each_iter = []
        for i in range(self.iteration_num):
            wandb.log({"iteration": i})
            if self.log:
                self.logger.info(
                    f"---------------------  iteration {i} ------------------------"
                )

            path, cum_rewards = self.iterate(self.root)
            self.trace_in_each_iter.append(deepcopy(path))

        mcts_output = self.prepare_output()
        self.output_to_json(mcts_output=mcts_output)
        return self.trace_in_each_iter, mcts_output

    def __call__(self, init_state: str, **kwargs):

        MCTSNode.reset_id()

        iteration_paths, mcts_outputs = self.search(init_state)

        return iteration_paths, mcts_outputs

    #################################################################################
    #                        Log and Evaluate Helper Functions                      #
    #################################################################################

    def eval_and_log_node(
        self, node: MCTSNode, eval=False, log_metric=False, eval_type="test"
    ):
        parent_info = (
            f"parent: {node.parent.id}" if node.parent is not None else "parent: N/A"
        )
        self.logger.info(
            f"node {node.id}:    {parent_info} | depth: {node.depth} | visited: {node.visited} | expand_times: {node.expand_times}  | terminal: {node.is_terminal} | children: {len(node.children)}"
        )
        self.logger.info(
            f"   reward: {node.reward:.4f} | Q: {node.Q:.4f} | uct: {self._uct(node):.4f} | cum_rewards: {node.cum_rewards}"
        )
        self.logger.info(f"   prompt: {node.prompt}")

        if eval:
            if eval_type == "test":
                test_metric, eval_output = self.world_model.test_prompt(node.prompt)
            else:
                raise ValueError(f"eval_type {eval_type} is not supported.")
            node.test_metric = test_metric
        if log_metric:
            if not isinstance(node.test_metric, tuple):
                self.logger.info(f"   {eval_type} metric: {node.test_metric:.4f}")
            else:
                self.logger.info(f"   {eval_type} metric: {node.test_metric}")
        self.logger.info("---------------------")
        if eval:
            return eval_output["correct"]
        else:
            return None

    def log_vars(self):
        self.logger.info("-------------------- MCTS -----------------------")
        ignored_print_vars = [
            "task",
            "log_dir",
            "logger",
            "trace_in_each_iter",
            "root",
            "nodes",
        ]
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars:
                continue
            var_value = vars_dict[var_name]
            self.logger.info(f"{var_name} : {var_value}")
        self.logger.info("-------------------------------------------")

    def log_path(self, path, eval=False, log_metric=False):
        for node in path:
            self.eval_and_log_node(node=node, eval=eval, log_metric=log_metric)

    def log_nodes(self, nodes, eval=False, log_metric=False, eval_type="test"):
        for i, node in enumerate(nodes):
            self.eval_and_log_node(
                node, eval=eval, log_metric=log_metric, eval_type=eval_type
            )
        self.logger.info("\n")

    def log_paths(self, paths, eval=False, log_metric=False, eval_type="test"):
        for i, path in enumerate(paths):
            self.logger.info(f"\n----------------  path {i} ------------------")
            for node in path:
                self.eval_and_log_node(
                    node, eval=eval, log_metric=log_metric, eval_type=eval_type
                )

    def _sort_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric

    def prepare_output(self):
        self.logger.info(
            "\n---------------------  all iteration paths ------------------------"
        )
        self.log_paths(self.trace_in_each_iter)
        self.logger.info("\n---------------------  all nodes ------------------------")
        self.log_nodes(self.nodes)

        # prepare output
        paths_nodes = []
        paths_ids = []
        paths_qs = []
        paths_rewards = []
        paths_ucts = []

        for i, path in enumerate(self.trace_in_each_iter):
            path_nodes = []
            path_ids = []
            path_qs = []
            path_rewards = []
            path_ucts = []

            for node in path:
                path_ids.append(node.id)
                uct = self._uct(node)
                node.uct = uct
                path_ucts.append(uct)
                path_nodes.append(node)
                path_qs.append(node.Q)
                path_rewards.append(node.reward)

            paths_nodes.append(path_nodes)
            paths_ids.append(path_ids)
            paths_qs.append(path_qs)
            paths_rewards.append(path_rewards)
            paths_ucts.append(path_ucts)

            self.logger.info(f"path {i}: {path_ids} ")
            self.logger.info(
                f"mean values:   path_uct: {np.mean(path_ucts):.4f} | path_q: {np.mean(path_qs):.4f} | path_reward: {np.mean(path_rewards):.4f}"
            )
            self.logger.info(f"path_ucts:  {path_ucts}")
            self.logger.info(f"paths_qs :  {paths_qs}")
            self.logger.info(f"path_reward : {path_rewards}")
            self.logger.info("---------------------------")

        qs_rank = sorted(
            range(len(paths_qs)), key=lambda i: np.mean(paths_qs[i]), reverse=True
        )
        rewards_rank = sorted(
            range(len(paths_rewards)),
            key=lambda i: np.mean(paths_rewards[i]),
            reverse=True,
        )

        best_q_path = paths_nodes[qs_rank[0]]
        best_reward_path = paths_nodes[rewards_rank[0]]
        top_k_reward_nodes = sorted(
            self.nodes, key=lambda node: node.reward, reverse=True
        )[: self.k]

        if len(self.world_model.test_dataloader) != 0:
            self.logger.info("\n----------------  test nodes ------------------")
            test_nodes_set = set(best_q_path + best_reward_path + top_k_reward_nodes)
            detailed_metrics_columns = []
            detailed_metrics_values = []
            if hasattr(self.task, "test_metrics_definition"):
                detailed_metrics_columns = ["node_id"] + [
                    f"{metric['metric_name']}_{suffix}"
                    for metric in self.task.test_metrics_definition
                    for suffix in ["YES", "NO", "NA"]
                ]
            for node in self.nodes:
                if node in test_nodes_set:
                    correct_results = np.array(
                        self.eval_and_log_node(
                            node, eval=True, log_metric=True, eval_type="test"
                        )
                    )
                    if len(correct_results.shape) == 2:
                        list_of_counts = self._record_counts(correct_results)
                        detailed_metrics_values.append([node.id] + list_of_counts)

            if len(detailed_metrics_values) > 0:
                wandb.log(
                    {
                        "detailed_metrics": wandb.Table(
                            columns=detailed_metrics_columns,
                            data=detailed_metrics_values,
                        )
                    }
                )
            self.logger.info("\n----------------  top_k_reward_nodes------------------")
            for node in top_k_reward_nodes:
                self.eval_and_log_node(
                    node, eval=False, log_metric=True, eval_type="test"
                )

            self.logger.info("\n----------------  best_reward_path------------------")
            for node in best_reward_path:
                self.eval_and_log_node(
                    node, eval=False, log_metric=True, eval_type="test"
                )

        selected_node = sorted(
            best_reward_path,
            key=lambda node: self._sort_helper(node.reward),
            reverse=True,
        )[0]

        last_node_of_best_reward_path = best_reward_path[-1]
        # log everything to wandb
        wandb.run.summary["test_accuracy"] = selected_node.test_metric  # type: ignore
        wandb.run.summary["last_node_test_accuracy"] = (  # type: ignore
            last_node_of_best_reward_path.test_metric
        )
        # make a table and send all the path data to wandb
        path_data_as_list = []
        path_data_columns = ["path_id"] + list(self.nodes[0].to_dict().keys())
        for i, path in enumerate(self.trace_in_each_iter):
            for node in path:
                node_dict = node.to_dict()
                if node_dict["parent"] == -1:
                    node_dict["parent"] = None
                path_data_as_list.append([i] + list(node_dict.values()))
        wandb.log(
            {"paths": wandb.Table(columns=path_data_columns, data=path_data_as_list)}
        )
        # make a table of all the nodes visited
        data_nodes_as_list = []
        data_nodes_columns = [
            key if key != "id" else "node_id" for key in self.nodes[0].to_dict().keys()
        ]
        data_nodes_columns.extend(["best_q_path", "best_reward_path", "selected_node"])
        best_reward_path_ids = [node.id for node in best_reward_path]
        best_q_path_ids = [node.id for node in best_q_path]
        for node in self.nodes:
            node_dict = node.to_dict()
            if node_dict["parent"] == -1:
                node_dict["parent"] = None
            data_nodes_as_list.append(
                list(node_dict.values())
                + [
                    node.id in best_q_path_ids,
                    node.id in best_reward_path_ids,
                    node.id == selected_node.id,
                ]
            )
        wandb_node_table = wandb.Table(
            columns=data_nodes_columns, data=data_nodes_as_list
        )
        tree_fields = {"node-id": "node_id", "node-parent": "parent"}
        tree = wandb.plot_table(
            vega_spec_name="rezazzr/tree_visualizer",
            data_table=wandb_node_table,
            fields=tree_fields,
        )
        wandb.log({"nodes": tree})
        # make a graph of the nodes and optim nodes all together
        data_optim_nodes_as_list = []
        data_optim_nodes_columns = list(self.optim_nodes[0].to_dict().keys())
        tree_fields = {"node-id": "node_id", "node-parent": "parent"}

        for node in self.optim_nodes:
            node_dict = node.to_dict()
            if node_dict["parent"] == -1:
                node_dict["parent"] = None
            data_optim_nodes_as_list.append(list(node_dict.values()))

        wandb_optim_node_table = wandb.Table(
            columns=data_optim_nodes_columns, data=data_optim_nodes_as_list
        )
        tree = wandb.plot_table(
            vega_spec_name="tree_optim_visualizer",
            data_table=wandb_optim_node_table,
            fields=tree_fields,
        )
        wandb.log({"optim_nodes": tree})

        # end of loging to wandb

        return dict(
            all_paths=paths_nodes,
            all_nodes=self.nodes,
            best_q_path=best_q_path,
            best_reward_path=best_reward_path,
            top_k_reward_nodes=top_k_reward_nodes,
            best_reward_path_last_node=[last_node_of_best_reward_path],
            best_reward_path_selected_node=[selected_node],
        )

    def output_to_json(self, mcts_output):
        data_to_save = {}
        paths = []
        for path in mcts_output["all_paths"]:
            paths.append([node.to_dict() for node in path])
        data_to_save["all_paths"] = paths

        for key in mcts_output:
            if key != "all_paths":
                data_to_save[key] = [node.to_dict() for node in mcts_output[key]]
        with open(os.path.join(self.log_dir, "data.json"), "w") as f:
            json.dump(data_to_save, f, indent=4)

    @staticmethod
    def _record_counts(array: np.ndarray) -> List[int]:
        counts_list = []

        # Iterate over each column
        for col in range(array.shape[1]):
            counts = {1: 0, 0: 0, -1: 0}  # Initialize the counts for each value
            unique, counts_array = np.unique(array[:, col], return_counts=True)
            counts.update(dict(zip(unique, counts_array)))
            # Extend the counts_list with the counts in the order -1, 0, 1
            counts_list.extend([counts[1], counts[0], counts[-1]])

        return counts_list
