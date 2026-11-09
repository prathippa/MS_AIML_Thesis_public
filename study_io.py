"""Atomic local study artifacts. Hashes detect accidental changes, not malicious edits."""

import json, hashlib, math, os, tempfile, datetime, platform
from pathlib import Path
from contextlib import contextmanager
from importlib.metadata import version, PackageNotFoundError


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def digest(x):
    return hashlib.sha256(
        json.dumps(
            clean(x),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def lines(p):
    return (
        [
            json.loads(x)
            for x in Path(p).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        if Path(p).exists()
        else []
    )


def atomic(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix="." + p.name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)
    finally:
        if Path(tmp).exists():
            os.unlink(tmp)


def write(p, x):
    atomic(
        p, json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_lines(p, x):
    atomic(
        p,
        "".join(
            json.dumps(clean(r), ensure_ascii=False, allow_nan=False) + "\n" for r in x
        ),
    )


def bind(p, x):
    if Path(p).exists():
        old = read(p)
        if old != {"sha256": digest(x), "contents": clean(x)}:
            raise ValueError(
                f"Frozen input/config changed: {p}; use a new experiment directory"
            )
    else:
        write(p, {"sha256": digest(x), "contents": x})


def verify_bound(p):
    x = read(p)
    if digest(x["contents"]) != x["sha256"]:
        raise ValueError(f"Artifact changed: {p}")
    return x["contents"]


def source_hashes():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def environment():
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-learn",
        "pytest",
        "anthropic",
        "groq",
        "openai",
        "python-dotenv",
        "matplotlib",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


@contextmanager
def lock(root):
    p = Path(root) / ".active.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(
            "Experiment locked; confirm no worker is running before removing a stale .active.lock"
        )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "time": utcnow()}, f)
        yield
    finally:
        p.unlink(missing_ok=True)
