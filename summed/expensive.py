import os
from typing import Any, Callable
import pickle

class Expensive:
    _save_directory = "compute_cache"
    abs_save_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), _save_directory)

    @classmethod
    def load_or_computeandsave(cls, object_id: str, compute_callable: Callable[[],Any], recompute: bool = False):
        # if no object_id provided, only compute it
        if not object_id:
            return compute_callable()
        # if file exists load it, else compute and save it
        file_path = os.path.join(cls.abs_save_directory, f"{object_id}.pkl")
        if os.path.exists(file_path) and not recompute:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            computed = compute_callable()
            with open(file_path, "wb") as f:
                pickle.dump(computed, f)
            return computed
