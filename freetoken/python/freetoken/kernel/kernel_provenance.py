"""Fail-closed provenance for prebuilt and JIT kernel shared objects.

Prebuilt kernels are looked up by name only, so an edited csrc file would keep
running the old .so. Every load hashes the module's recipe (source files with
line endings normalized, the csrc/include tree, wrappers, flags) and accepts a
prebuilt .so only when a manifest binds that exact digest to that exact .so.

Stdlib only: release tooling can load this file by path without importing
freetoken (``inventory`` / ``check`` never touch CUDA).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import pathlib
import time
from typing import Any, Iterable

SCHEMA = 1
KERNEL_DIR = pathlib.Path(__file__).resolve().parent
MANIFEST_NAME = "kernel_cache_manifest.json"
# written next to the .so files by a provenance-aware cache build (aot.compile_and_package_kernels)
CACHE_PROVENANCE_NAME = "freetoken_provenance.json"
RECORD_NAME = "freetoken_provenance_record.json"
PRIVATE_JIT_DIR_ENV = "FREETOKEN_PROVENANCE_JIT_DIR"
# setuptools extensions shipped as prebuilt .so inside the freetoken package
EXTENSIONS = {
    "_cpu_moe": ("cpu_moe/cpu_moe_ext.cpp",),
    "_pinned_tensor": ("pinned_tensor.cpp",),
    "_ple_store": ("ple_store/ple_store_ext.cpp",),
}
OK_MODULE_STATUSES = ("prebuilt", "jit-private-built")
_LOADED: dict[str, dict[str, Any]] = {}


class ProvenanceError(RuntimeError):
    pass


def text_sha256(path: str | os.PathLike[str]) -> str:
    """sha256 of a source file with CRLF folded to LF (compilers treat both alike; git stores LF)."""
    return hashlib.sha256(pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rel(path: str | os.PathLike[str], csrc: pathlib.Path) -> str:
    p = pathlib.Path(path).resolve()
    try:
        return p.relative_to(csrc.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _include_tree(csrc: pathlib.Path) -> dict[str, str]:
    root = csrc / "include"
    if not root.is_dir():
        return {}
    return {p.relative_to(csrc).as_posix(): text_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def make_recipe(
    name: str,
    loader: str,
    sources: Iterable[str | os.PathLike[str]],
    wrappers: Iterable[tuple[str, str]] = (),
    flags: dict[str, list[str]] | None = None,
    csrc: pathlib.Path | None = None,
) -> dict[str, Any]:
    csrc = csrc or KERNEL_DIR / "csrc"
    return {
        "schema": SCHEMA,
        "name": name,
        "loader": loader,
        "sources": {_rel(p, csrc): text_sha256(p) for p in sources},
        # every header under csrc/include counts: a header edit invalidates all modules (conservative)
        "include": _include_tree(csrc),
        "wrappers": [list(w) for w in wrappers],
        "flags": {k: list(v) for k, v in sorted((flags or {}).items())},
    }


def recipe_digest(recipe: dict[str, Any]) -> str:
    return _digest(recipe)


def refresh_recipe(recipe: dict[str, Any], csrc: pathlib.Path) -> dict[str, Any]:
    """The same recipe with every source re-hashed from ``csrc`` (missing file -> None)."""
    out = dict(recipe)

    def h(rel: str) -> str | None:
        p = csrc / rel
        return text_sha256(p) if p.is_file() else None

    out["sources"] = {rel: h(rel) for rel in recipe["sources"]}
    out["include"] = _include_tree(csrc)
    return out


def changed_sources(recipe: dict[str, Any], reference: dict[str, str]) -> list[str]:
    files = {**recipe.get("sources", {}), **recipe.get("include", {})}
    return sorted(rel for rel, sha in files.items() if reference.get(rel) != sha)


# ---------------------------------------------------------------------------- manifests


_MANIFEST_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def load_manifest(path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    path = pathlib.Path(path) if path else KERNEL_DIR / MANIFEST_NAME
    if not path.is_file():
        return None
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _MANIFEST_CACHE.get(str(path))
    if hit is not None and hit[0] == key:
        return hit[1]
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise ProvenanceError(f"{path}: kernel provenance manifest schema {data.get('schema')!r}, expected {SCHEMA}")
    data["_path"] = str(path)
    data["_sha256"] = file_sha256(path)
    _MANIFEST_CACHE[str(path)] = (key, data)
    return data


def load_cache_provenance(cache_dir: pathlib.Path | None) -> dict[str, Any]:
    if cache_dir is None:
        return {}
    path = pathlib.Path(cache_dir) / CACHE_PROVENANCE_NAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return data.get("modules", {}) if data.get("schema") == SCHEMA else {}


def lookup(name: str, manifest: dict[str, Any] | None, cache_dir: pathlib.Path | None):
    """(entry, origin) for a prebuilt module: the in-tree manifest wins over a cache-shipped record."""
    if manifest and name in manifest.get("modules", {}):
        return manifest["modules"][name], "manifest"
    entry = load_cache_provenance(cache_dir).get(name)
    return (entry, "cache-provenance") if entry else (None, None)


def decide_prebuilt(
    name: str,
    recipe: dict[str, Any],
    so_path: pathlib.Path,
    *,
    manifest: dict[str, Any] | None,
    cache_dir: pathlib.Path | None,
    strict: bool = False,
) -> dict[str, Any]:
    """Whether the prebuilt ``so_path`` may stand in for ``recipe``; raises on a replaced .so."""
    digest = recipe_digest(recipe)
    if strict and (manifest is None or name not in manifest.get("modules", {})):
        return {"use": False, "reason": "missing-manifest" if manifest is None else "unknown-module", "recipe_digest": digest}
    entry, origin = lookup(name, manifest, cache_dir)
    if entry is not None:
        if strict and (not isinstance(entry, dict) or not isinstance(entry.get("recipe"), dict)
                       or recipe_digest(entry["recipe"]) != entry.get("recipe_digest")):
            raise ProvenanceError(f"invalid manifest recipe for {name!r}")
        if entry["recipe_digest"] != digest:
            return {"use": False, "reason": "recipe-changed", "recipe_digest": digest,
                    "manifest_digest": entry["recipe_digest"],
                    "changed": changed_sources(recipe, {**entry["recipe"]["sources"], **entry["recipe"]["include"]})}
        so_sha = file_sha256(so_path)
        if so_sha != entry["so_sha256"]:
            raise ProvenanceError(
                f"prebuilt kernel {name!r} at {so_path} has sha256 {so_sha}, but the {origin} records "
                f"{entry['so_sha256']} for this recipe: the .so was replaced or corrupted; refusing to load it"
            )
        return {"use": True, "provenance": f"prebuilt:{origin}", "recipe_digest": digest, "so_sha256": so_sha}
    upstream = (manifest or {}).get("upstream", {}).get("csrc", {})
    changed = changed_sources(recipe, upstream) if upstream else ["<no manifest>"]
    if not changed:
        return {"use": True, "provenance": "prebuilt:upstream-unverified", "recipe_digest": digest,
                "so_sha256": file_sha256(so_path)}
    return {"use": False, "reason": "unknown-module", "recipe_digest": digest, "changed": changed}


# ---------------------------------------------------------------------------- private JIT


def private_root() -> pathlib.Path:
    raw = os.getenv(PRIVATE_JIT_DIR_ENV)
    return pathlib.Path(raw).expanduser() if raw else pathlib.Path("~/.cache/freetoken/provenance-jit").expanduser()


def private_build_dir(name: str, digest: str) -> pathlib.Path:
    return private_root() / name / digest[:24]


def find_built_so(build_dir: pathlib.Path, name: str) -> pathlib.Path | None:
    found = [p for p in pathlib.Path(build_dir).rglob(f"{name}.so")]
    return max(found, key=lambda p: p.stat().st_mtime_ns) if found else None


@contextlib.contextmanager
def build_lock(build_dir: pathlib.Path):
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # non-POSIX: no concurrent builders expected
        yield
        return
    with open(build_dir / ".provenance.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_record(build_dir: pathlib.Path, name: str, recipe: dict[str, Any], so_path: pathlib.Path) -> dict[str, Any]:
    record = {"schema": SCHEMA, "name": name, "recipe_digest": recipe_digest(recipe), "recipe": recipe,
              "so_path": str(so_path), "so_sha256": file_sha256(so_path),
              "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = build_dir / (RECORD_NAME + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True))
    tmp.replace(build_dir / RECORD_NAME)
    return record


def read_record(build_dir: pathlib.Path) -> dict[str, Any] | None:
    path = pathlib.Path(build_dir) / RECORD_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def note_loaded(name: str, info: dict[str, Any]) -> None:
    _LOADED[name] = dict(info)


def loaded_modules() -> dict[str, dict[str, Any]]:
    """What this process actually loaded: name -> provenance, so path, sha256s."""
    return {k: dict(v) for k, v in _LOADED.items()}


# ---------------------------------------------------------------------------- inventory / check


def default_cache_dir() -> pathlib.Path | None:
    """The cache dir the runtime would use (same env rules as kernel/utils.py), without importing freetoken."""
    if os.getenv("FREETOKEN_DISABLE_KERNEL_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    override = os.getenv("FREETOKEN_KERNEL_CACHE_DIR")
    if override:
        return pathlib.Path(override).expanduser()
    spec = importlib.util.find_spec("freetoken_kernel_cache")
    if spec is None or not spec.origin:
        return None
    return pathlib.Path(spec.origin).parent / "jit_cache"


def _extension_so(kernel_dir: pathlib.Path, ext: str) -> pathlib.Path | None:
    found = sorted(kernel_dir.glob(f"{ext}.*.so")) + sorted(kernel_dir.glob(f"{ext}.so"))
    return found[0] if found else None


def extension_sources(kernel_dir: pathlib.Path, ext: str) -> dict[str, str | None]:
    csrc = kernel_dir / "csrc"
    return {rel: (text_sha256(csrc / rel) if (csrc / rel).is_file() else None) for rel in EXTENSIONS[ext]}


def inventory(
    *,
    kernel_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    use_default_cache: bool = True,
) -> dict[str, Any]:
    """CPU-only report of every prebuilt kernel module and extension: what the runtime would load and why.

    Module status: ``prebuilt`` (manifest digest == current sources, .so sha256 == manifest),
    ``jit-private-built`` / ``jit-pending`` (sources changed: the prebuilt is refused and a private
    JIT build is / is not present), ``missing-so``, ``so-mismatch`` (replaced .so: load raises),
    ``unknown`` (in the cache dir but not in any manifest). Extension status: ``prebuilt``,
    ``stale-source`` (csrc edited after the .so was built: the .so does not run the current
    source), ``so-mismatch`` (.so differs from the manifest), ``unknown``, ``absent``.
    """
    kdir = pathlib.Path(kernel_dir).resolve() if kernel_dir else KERNEL_DIR
    csrc = kdir / "csrc"
    manifest = load_manifest(manifest_path or kdir / MANIFEST_NAME)
    cdir = pathlib.Path(cache_dir) if cache_dir else (default_cache_dir() if use_default_cache else None)
    cache_prov = load_cache_provenance(cdir)
    names = set((manifest or {}).get("modules", {})) | set(cache_prov)
    if cdir is not None and cdir.is_dir():
        names |= {p.name for p in cdir.iterdir() if (p / f"{p.name}.so").is_file()}
    modules = []
    for name in sorted(names):
        entry, origin = lookup(name, manifest, cdir)
        so = cdir / name / f"{name}.so" if cdir is not None else None
        so_sha = file_sha256(so) if so is not None and so.is_file() else None
        row: dict[str, Any] = {"name": name, "origin": origin, "so_path": str(so) if so else None, "so_sha256": so_sha}
        if entry is None:
            row["status"] = "unknown"
            modules.append(row)
            continue
        current = refresh_recipe(entry["recipe"], csrc)
        digest = recipe_digest(current)
        row.update(manifest_so_sha256=entry["so_sha256"], manifest_recipe_digest=entry["recipe_digest"],
                   source_sha256=digest, sources=current["sources"])
        if digest != entry["recipe_digest"]:
            pdir = private_build_dir(name, digest)
            rec = read_record(pdir)
            built = rec is not None and rec.get("recipe_digest") == digest and pathlib.Path(rec["so_path"]).is_file() \
                and file_sha256(rec["so_path"]) == rec["so_sha256"]
            row.update(status="jit-private-built" if built else "jit-pending", private_dir=str(pdir),
                       changed=changed_sources(current, {**entry["recipe"]["sources"], **entry["recipe"]["include"]}))
            if built:
                row.update(jit_so_path=rec["so_path"], jit_so_sha256=rec["so_sha256"])
        elif so_sha is None:
            row["status"] = "missing-so"
        elif so_sha != entry["so_sha256"]:
            row["status"] = "so-mismatch"
        else:
            row["status"] = "prebuilt"
        modules.append(row)
    extensions = []
    for ext in EXTENSIONS:
        so = _extension_so(kdir, ext)
        srcs = extension_sources(kdir, ext)
        ment = (manifest or {}).get("extensions", {}).get(ext)
        row = {"name": ext, "so_path": str(so) if so else None, "so_sha256": file_sha256(so) if so else None,
               "source_sha256": _digest(srcs), "sources": srcs}
        if so is None:
            row["status"] = "absent"
        elif ment is None:
            row["status"] = "unknown"
        elif row["so_sha256"] != ment["so_sha256"]:
            row["status"] = "so-mismatch"
        elif row["source_sha256"] != ment["source_sha256"]:
            row["status"] = "stale-source"
        else:
            row["status"] = "prebuilt"
        if ment is not None:
            row.update(manifest_so_sha256=ment["so_sha256"], manifest_source_sha256=ment["source_sha256"])
        extensions.append(row)
    return {
        "schema": SCHEMA,
        "kernel_dir": str(kdir),
        "cache_dir": str(cdir) if cdir is not None else None,
        "manifest": None if manifest is None else {"path": manifest["_path"], "sha256": manifest["_sha256"],
                                                   "basis": manifest.get("basis")},
        "modules": modules,
        "extensions": extensions,
    }


def check(
    inv: dict[str, Any] | None = None,
    *,
    required_extensions: Iterable[str] = ("_pinned_tensor",),
    allow_module_statuses: Iterable[str] = OK_MODULE_STATUSES,
    allow_unknown_modules: bool = False,
    **inventory_kwargs: Any,
) -> dict[str, Any]:
    """Raise ProvenanceError unless every module and every required extension is provably current.

    ``required_extensions``: the setuptools extensions the selected profile runs (``_cpu_moe`` only
    for --moe-strategy cpu/hybrid). Returns the inventory on success.
    """
    inv = inv if inv is not None else inventory(**inventory_kwargs)
    allowed = set(allow_module_statuses) | ({"unknown"} if allow_unknown_modules else set())
    problems = []
    if inv["manifest"] is None:
        problems.append("no kernel provenance manifest")
    for row in inv["modules"]:
        if row["status"] not in allowed:
            problems.append(f"kernel {row['name']}: {row['status']}")
    required = set(required_extensions)
    for row in inv["extensions"]:
        if row["name"] in required and row["status"] != "prebuilt":
            problems.append(f"extension {row['name']}: {row['status']}")
    if problems:
        raise ProvenanceError("kernel provenance check failed: " + "; ".join(problems))
    return inv


# ---------------------------------------------------------------------------- manifest generation


def generate_manifest(
    *,
    recipes: dict[str, dict[str, Any]],
    cache_dir: pathlib.Path,
    upstream_csrc: dict[str, str],
    upstream_ref: str,
    basis: dict[str, Any],
    kernel_dir: pathlib.Path | None = None,
    extension_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Bind each cached .so to the recipe it was built from.

    A module is recorded only when its recipe sources equal ``upstream_csrc`` (the tree the
    release cache was built from); others are left out and so stay ``unknown`` (fail-closed).
    """
    kdir = kernel_dir or KERNEL_DIR
    modules, skipped = {}, {}
    for name, recipe in sorted(recipes.items()):
        so = cache_dir / name / f"{name}.so"
        changed = changed_sources(recipe, upstream_csrc)
        if not so.is_file():
            skipped[name] = "no .so in cache"
        elif changed:
            skipped[name] = f"sources differ from {upstream_ref}: {changed}"
        else:
            modules[name] = {"recipe_digest": recipe_digest(recipe), "recipe": recipe,
                             "so_sha256": file_sha256(so), "so_size": so.stat().st_size}
    extensions = {}
    for ext, rels in EXTENSIONS.items():
        so = _extension_so(extension_dir or kdir, ext)
        srcs = {rel: upstream_csrc.get(rel) for rel in rels}
        if so is not None and all(srcs.values()):
            extensions[ext] = {"sources": srcs, "source_sha256": _digest(srcs), "so_sha256": file_sha256(so),
                               "so_name": so.name}
    return {"schema": SCHEMA, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "basis": basis, "upstream": {"ref": upstream_ref, "csrc": dict(sorted(upstream_csrc.items()))},
            "modules": modules, "extensions": extensions, "skipped": skipped}


def record_extension_build(manifest_path: str | os.PathLike[str], ext: str, kernel_dir: str | os.PathLike[str],
                           basis: str) -> dict[str, Any]:
    """After rebuilding an extension from the current csrc (``setup.py build_ext``), bind its new .so
    to the current sources in the manifest; ``basis`` says who built it and how."""
    path = pathlib.Path(manifest_path)
    data = json.loads(path.read_text())
    kdir = pathlib.Path(kernel_dir)
    so = _extension_so(kdir, ext)
    if so is None:
        raise ProvenanceError(f"no built {ext} .so under {kdir}")
    srcs = extension_sources(kdir, ext)
    if not all(srcs.values()):
        raise ProvenanceError(f"{ext}: missing sources {srcs}")
    data.setdefault("extensions", {})[ext] = {"sources": srcs, "source_sha256": _digest(srcs),
                                              "so_sha256": file_sha256(so), "so_name": so.name, "basis": basis}
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data["extensions"][ext]


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="kernel provenance inventory/check (CPU only)")
    ap.add_argument("command", choices=("inventory", "check", "record-extension"))
    ap.add_argument("--kernel-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--manifest")
    ap.add_argument("--require-extension", action="append", default=None)
    ap.add_argument("--extension")
    ap.add_argument("--basis")
    args = ap.parse_args(argv)
    if args.command == "record-extension":
        if not (args.manifest and args.extension and args.kernel_dir and args.basis):
            ap.error("record-extension needs --manifest --extension --kernel-dir --basis")
        print(json.dumps(record_extension_build(args.manifest, args.extension, args.kernel_dir, args.basis), indent=1))
        return 0
    kw = dict(kernel_dir=args.kernel_dir, cache_dir=args.cache_dir, manifest_path=args.manifest)
    inv = inventory(**kw)
    if args.command == "check":
        try:
            check(inv, required_extensions=args.require_extension or ("_pinned_tensor",))
        except ProvenanceError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 1
    print(json.dumps(inv, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
