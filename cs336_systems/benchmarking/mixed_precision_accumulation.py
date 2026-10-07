"""Problem ``mixed_precision_accumulation``: accumulate 1000 x 0.01 in different precisions.

uv run python -m cs336_systems.benchmarking.mixed_precision_accumulation
"""

import torch


def main() -> None:
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print("fp32 accumulator, fp32 addends:        ", s.item())

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp16 accumulator, fp16 addends:        ", s.item())

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp32 accumulator, fp16 addends:        ", s.item())

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print("fp32 accumulator, fp16 addends upcast: ", s.item())


if __name__ == "__main__":
    main()
