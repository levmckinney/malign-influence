from datasets import Dataset
from itertools import product
import random

def get_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"input": f"{x1}+{x2}", "target": f"{(x1 + x2) % 10}"}
            for x1, x2 in product(range(10), range(10)) if random.random() < 0.95
        ]
    )
