import sys
import time

start = time.time()
for line in sys.stdin:
    sys.stdout.write(f"[{time.time() - start:7.1f}s] {line}")
    sys.stdout.flush()
