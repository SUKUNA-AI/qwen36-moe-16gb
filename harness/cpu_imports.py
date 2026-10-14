"""Test-only namespace packages: use real CPU modules, avoid eager model initializers.

No kernel operation or state/cache implementation is mocked by this bootstrap.
Do not use it for inference or qualification.
"""
import sys
import atexit
import shutil
import tempfile
import types
from repo_paths import FREETOKEN

def enable():
    for name in ('engine', 'scheduler', 'models', 'models.qwen3_5_moe', 'layers', 'kernel.fla'):
        full = 'freetoken.' + name
        if full not in sys.modules:
            package = types.ModuleType(full)
            package.__path__ = [str(FREETOKEN/'python/freetoken'/name.replace('.', '/'))]
            sys.modules[full] = package

    # tvm-ffi 0.1.13 writes unescaped source paths into Ninja files. In a checkout
    # with spaces, compile the identical CPU radix source/headers from a temporary
    # path. Recipe source/header hashes stay identical; no arithmetic is replaced.
    if any(ch.isspace() for ch in str(FREETOKEN)):
        from freetoken.kernel import utils
        temp = tempfile.TemporaryDirectory(prefix='qwen36-cpu-csrc-', dir='/tmp')
        atexit.register(temp.cleanup)
        target = __import__('pathlib').Path(temp.name)/'csrc'
        source = FREETOKEN/'python/freetoken/kernel/csrc'
        shutil.copytree(source/'include',target/'include')
        (target/'src').mkdir()
        shutil.copyfile(source/'src/radix.cpp',target/'src/radix.cpp')
        utils.KERNEL_PATH=target
        utils.DEFAULT_INCLUDE=[str(target/'include')]
