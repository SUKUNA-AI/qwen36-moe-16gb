"""Source-checkout paths, independent of CWD; native runtime paths stay explicit."""
import os
from pathlib import Path

CHECKOUT = Path(os.environ.get('QWEN36_CHECKOUT', Path(__file__).resolve().parents[1])).resolve()
SERVING = Path(os.environ.get('QWEN36_SERVING_SOURCE', CHECKOUT / 'serving')).resolve()
HARNESS = Path(os.environ.get('QWEN36_HARNESS_SOURCE', CHECKOUT / 'harness')).resolve()
FREETOKEN = Path(os.environ.get('QWEN36_FREETOKEN_SOURCE', CHECKOUT / 'freetoken')).resolve()
KERNELS = Path(os.environ.get('QWEN36_KERNELS_SOURCE', CHECKOUT / 'kernels')).resolve()
