"""Проверка: сколько памяти можно выделить в одном процессе и в нескольких одновременно."""
import sys
from multiprocessing import Pool

import numpy as np


def grab(gb: int) -> str:
    L = []
    for i in range(gb):
        try:
            L.append(np.ones(2**27))  # 1 ГБ
        except MemoryError:
            return f"FAIL at {i + 1} GB"
    return f"ok {gb} GB"


if __name__ == "__main__":
    print("один процесс:", grab(int(sys.argv[1])), flush=True)
    with Pool(int(sys.argv[2])) as p:
        print("параллельно:", p.map(grab, [1] * int(sys.argv[2])), flush=True)
