import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AUTOEMBED_ROOT", str(ROOT))
for p in (str(ROOT), str(ROOT / "agent_task")):
    if p not in sys.path:
        sys.path.insert(0, p)
