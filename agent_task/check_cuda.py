import sys

import torch


count = torch.cuda.device_count()
if not torch.cuda.is_available() or count != 1:
    print(f"expected exactly one visible CUDA GPU, found {count}", file=sys.stderr)
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
print(
    f"one CUDA GPU visible: logical=0 name={props.name} "
    f"memory={props.total_memory // 2**20}MiB"
)
