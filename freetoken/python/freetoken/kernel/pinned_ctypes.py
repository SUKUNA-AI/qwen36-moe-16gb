"""Load a private ctypes library only if it is exactly the pinned build (shared by the opt-in decode
kernels cuda_oproj_gemv and cuda_moe_decode; same scheme as cuda_fp8_gemv):

  * the file is read once (O_NOFOLLOW, regular file, bounded size) and must hash to the pinned
    sha256; the verified bytes are dlopen'ed from an anonymous sealed memfd, so no path is
    re-opened between the hash and the load;
  * the caller then probes constant entry points (ABI version, compiled-in source sha256) before
    any CUDA call.
Any failure raises ``PinnedLibraryError``: an explicitly requested library that is not exactly the
pinned build is a deployment error, never silently loaded.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat

_MAX_BYTES = 64 << 20
# Linux memfd sealing ABI (some Python builds' fcntl module lacks the names; the values are stable)
_MFD_ALLOW_SEALING = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
_F_ADD_SEALS = 1024 + 9
_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008  # F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE


class PinnedLibraryError(RuntimeError):
    """The requested library is not exactly the pinned build; it was not loaded."""


def read_pinned(path: str | None, sha256: str, env_name: str) -> bytes:
    """The library bytes, only if they hash to ``sha256`` (read once, never through a symlink)."""
    if not path or not os.path.isabs(path):
        raise PinnedLibraryError(f"{env_name} must be the absolute path of the pinned library (sha256 {sha256}), "
                                 f"got {path!r}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PinnedLibraryError(f"cannot open {env_name}={path}: {exc.strerror}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            raise PinnedLibraryError(f"{env_name}={path} is not a regular file of at most {_MAX_BYTES} bytes")
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise PinnedLibraryError(f"{env_name}={path} has sha256 {digest}, but the loader pins {sha256}: a stale or "
                                 "foreign build; refusing to load it")
    return data


def dlopen_verified(data: bytes, name: str) -> tuple[ctypes.CDLL, int]:
    """dlopen exactly the hashed bytes, from an anonymous sealed memfd (no path is re-opened)."""
    import fcntl

    fd = os.memfd_create(name, os.MFD_CLOEXEC | _MFD_ALLOW_SEALING)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        try:
            fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", _F_ADD_SEALS), _SEALS)
        except OSError:
            pass  # seals are defence in depth: the bytes were hashed in this process before the write
        lib = ctypes.CDLL(f"/proc/self/fd/{fd}")
    except BaseException:
        os.close(fd)
        raise
    return lib, fd


def constant(lib: ctypes.CDLL, name: str, restype):
    """Value of a constant-returning entry point (no CUDA call); PinnedLibraryError if it is missing."""
    fn = getattr(lib, name, None)
    if fn is None:
        raise PinnedLibraryError(f"library has no {name}(): not the pinned ABI")
    fn.argtypes = []
    fn.restype = restype
    value = fn()
    if restype is ctypes.c_char_p:
        return value.decode("ascii", "replace") if value else None
    return value


def load(path: str | None, sha256: str, env_name: str, memfd_name: str) -> tuple[ctypes.CDLL, int]:
    """read_pinned + dlopen_verified; OSError from the dlopen becomes PinnedLibraryError."""
    data = read_pinned(path, sha256, env_name)
    try:
        return dlopen_verified(data, memfd_name)
    except OSError as exc:
        raise PinnedLibraryError(f"cannot load the verified {env_name} bytes: {exc}") from None
