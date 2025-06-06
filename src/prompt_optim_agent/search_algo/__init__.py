from . import *
from .mcts import MCTS

SEARCH_ALGOS = {"mcts": MCTS}


def get_search_algo(algo_name):
    assert (
        algo_name in SEARCH_ALGOS.keys()
    ), f"Search algo {algo_name} is not supported."
    return SEARCH_ALGOS[algo_name]
