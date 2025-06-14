import os.path
import pickle
from pathlib import Path
from typing import Union

path_to_here = os.path.abspath(os.path.dirname(__file__))
path_to_data = os.path.abspath(f"{path_to_here}/../../../data")


def save_pickle(obj: object, file_path: Union[str, Path]) -> None:
    """Save an object to disk using pickle."""
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)


def load_pickle(file_path: Union[str, Path]) -> object:
    """Load an object from disk using pickle."""
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)
