"""Discover compatible local models without copying weights or downloading them."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


DEFAULT_MODEL = "large-v3-turbo"
MODEL_NAMES = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium",
               "medium.en", "large-v1", "large-v2", "large-v3", DEFAULT_MODEL)
BACKENDS = ("auto", "mlx", "faster-whisper", "whisper", "whisper-cpp", "macwhisper")


@dataclass(frozen=True)
class ModelSource:
    backend: str
    location: str
    name: str
    origin: str
    cached: bool = True


def canonical_model(name: str) -> str:
    return {"turbo": DEFAULT_MODEL, "large": "large-v3"}.get(name, name)


def model_cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "srt-to-premiere" / "models"


def cache_roots(extra: list[str] | None = None) -> list[tuple[str, Path]]:
    """Known application caches only; never scan the user's entire home directory."""
    home = Path.home()
    cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    roots = [("srt-to-premiere" if Path(value).expanduser() == model_cache_dir() else "custom",
              Path(value).expanduser()) for value in extra or []]
    roots += [("srt-to-premiere", model_cache_dir()), ("Whisper", cache / "whisper")]
    hf_home = Path(os.environ.get("HF_HOME", cache / "huggingface"))
    roots.append(("Hugging Face", Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub")))))
    if os.environ.get("BUZZ_MODEL_ROOT"):
        roots.append(("Buzz", Path(os.environ["BUZZ_MODEL_ROOT"]).expanduser()))
    if platform.system() == "Darwin":
        roots += [("Buzz", home / "Library/Caches/Buzz/models"),
                  ("MacWhisper", home / "Library/Application Support/MacWhisper/models"),
                  ("MacWhisper", home / "Library/Containers/com.goodsnooze.MacWhisper/Data/Library/Application Support/MacWhisper/models")]
    elif platform.system() == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local"))
        roots += [("Buzz", local / "Buzz/Buzz/Cache/models"), ("Buzz", local / "Buzz/Cache/models")]
    else:
        roots.append(("Buzz", cache / "Buzz/models"))
    unique = {}
    for origin, root in roots:
        unique.setdefault(str(root), (origin, root))
    return list(unique.values())


def faster_repo(name: str) -> str:
    name = canonical_model(name)
    return ("mobiuslabsgmbh/faster-whisper-large-v3-turbo" if name == DEFAULT_MODEL
            else f"Systran/faster-whisper-{name}" if name in MODEL_NAMES else name)


def is_ct2_model(path: Path) -> bool:
    # Including the tokenizer avoids a hidden tokenizer download in offline mode.
    return all((path / filename).is_file() for filename in ("model.bin", "config.json", "tokenizer.json"))


def mlx_available() -> bool:
    # Checking package presence keeps model listing independent of heavy runtimes.
    return (platform.system() == "Darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx_whisper") is not None)


def is_mlx_model(path: Path) -> bool:
    return ((path / "config.json").is_file()
            and any((path / filename).is_file() for filename in ("weights.safetensors", "weights.npz")))


def converted_model_path(checkpoint: Path, name: str, cache: Path) -> Path:
    stat = checkpoint.stat()
    identity = f"v1:{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return cache / "mlx" / f"{name}-{fingerprint}"


def macwhisper_models(executable: str | None = None) -> list[ModelSource]:
    executable = executable or shutil.which("mw")
    if not executable:
        return []
    try:
        result = subprocess.run([executable, "models", "list"], capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    models = []
    for line in result.stdout.splitlines():
        # Only installed local Whisper engines; never select a cloud provider.
        match = re.search(r"\b((?:whisperkit|whisper-cpp):\S+)", line)
        if match:
            identifier = match.group(1)
            name = identifier.split(":", 1)[1]
            for prefix in ("openai_whisper-", "ggml-model-whisper-", "ggml-"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            name = re.sub(r"-q\d_[01]$", "", name).replace("_turbo", "-turbo")
            models.append(ModelSource("macwhisper", identifier, canonical_model(name), "MacWhisper CLI"))
    return models


def discover_models(name: str, roots: list[tuple[str, Path]] | None = None,
                    app_models: list[ModelSource] | None = None) -> list[ModelSource]:
    name = canonical_model(name)
    models = []
    for origin, root in cache_roots() if roots is None else roots:
        if not root.is_dir():
            continue
        # Prefer the cache's current snapshot, not a stale older revision.
        repository = root / ("models--" + faster_repo(name).replace("/", "--"))
        revision_file = repository / "refs/main"
        try:
            revisions = [revision_file.read_text().strip()] if revision_file.is_file() else []
            snapshots = [repository / "snapshots" / revision for revision in revisions]
            snapshots += sorted((repository / "snapshots").glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
            direct = [root / name, root / f"faster-whisper-{name}"]
            for snapshot in direct + snapshots:
                if is_ct2_model(snapshot):
                    models.append(ModelSource("faster-whisper", str(snapshot), name, origin))
                elif is_mlx_model(snapshot):
                    models.append(ModelSource("mlx", str(snapshot), name, origin))
            for converted in sorted((root / "mlx").glob(f"{name}-*")):
                metadata = converted / "source.json"
                if not is_mlx_model(converted) or not metadata.is_file():
                    continue
                try:
                    data = json.loads(metadata.read_text(encoding="utf-8"))
                    if not isinstance(data, dict) or data.get("model") != name:
                        continue  # large-v3 must never match large-v3-turbo.
                    original = Path(data["source"])
                    if original.exists() and converted != converted_model_path(original, name, root):
                        continue  # The original checkpoint changed; rebuild its cache.
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                models.append(ModelSource("mlx", str(converted), name, origin))
            if name in MODEL_NAMES:
                for checkpoint in (root / f"{name}.pt", root / "whisper" / f"{name}.pt"):
                    if checkpoint.is_file():
                        models.append(ModelSource("whisper", str(checkpoint), name, origin))
                pattern = re.compile(rf"^(?:ggml-|ggml-model-whisper-)?{re.escape(name)}(?:-q\d_[01])?\.bin$")
                for directory in (root, root / "whisper-cpp", root / "whispercpp"):
                    for checkpoint in sorted(directory.glob("*.bin")):
                        if pattern.fullmatch(checkpoint.name):
                            models.append(ModelSource("whisper-cpp", str(checkpoint), name, origin))
        except (OSError, ValueError, KeyError):
            # An unreadable app cache must not prevent checking other caches.
            continue
    models += [source for source in app_models or [] if source.name == name or source.location == name]
    unique = {}
    for source in models:
        unique.setdefault((source.backend, source.location), source)
    return list(unique.values())


def resolve_model(name: str = DEFAULT_MODEL, backend: str = "auto", *, offline: bool = False,
                  roots: list[tuple[str, Path]] | None = None,
                  cpp_executable: str | None = None,
                  app_models: list[ModelSource] | None = None,
                  device: str = "auto") -> ModelSource:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    name = canonical_model(name)
    use_mlx = mlx_available() and device in ("auto", "metal")
    if backend == "mlx" and not use_mlx:
        raise ValueError("MLX requires Apple Silicon, --device auto/metal, and mlx-whisper (run uv sync)")
    path = Path(name).expanduser()
    if path.exists():
        if is_mlx_model(path):
            detected = "mlx"
        elif is_ct2_model(path):
            detected = "faster-whisper"
        elif path.is_file() and path.suffix == ".pt":
            detected = "mlx" if use_mlx and backend in ("auto", "mlx") else "whisper"
        elif path.is_file() and path.suffix == ".bin":
            detected = "whisper-cpp"
        else:
            raise ValueError("Local model must be a Whisper .pt, a whisper.cpp .bin, or a complete MLX/CTranslate2 directory")
        if backend not in ("auto", detected):
            raise ValueError(f"This model requires --backend {detected}")
        return ModelSource(detected, str(path.resolve()), canonical_model(path.stem), "explicit path")
    if name.startswith(("/", "~", ".")) or name.endswith((".pt", ".bin")):
        raise ValueError(f"Local model does not exist: {name}")
    candidates = discover_models(name, roots, app_models)
    if use_mlx and backend in ("auto", "mlx"):
        # Convert matching PyTorch weights locally once, without downloading them.
        candidates += [ModelSource("mlx", source.location, source.name, source.origin)
                       for source in candidates if source.backend == "whisper"]
    devices = {"mlx": ("auto", "metal"), "whisper-cpp": ("auto", "cpu", "metal"),
               "macwhisper": ("auto",), "whisper": ("auto", "cpu", "cuda"),
               "faster-whisper": ("auto", "cpu", "cuda")}
    allowed = [source for source in candidates if backend in ("auto", source.backend)
               and device in devices[source.backend]]
    cpp_executable = cpp_executable or shutil.which("whisper-cli")
    # Cached native engines first on Apple Silicon; CT2 is optimized for CPU/CUDA.
    preference = (("mlx",) if use_mlx else ()) + (("whisper-cpp", "macwhisper", "faster-whisper", "whisper") if platform.system() == "Darwin" else ("faster-whisper", "whisper-cpp", "whisper", "macwhisper"))
    for engine in preference:
        for source in allowed:
            if source.backend == engine and (engine != "whisper-cpp" or cpp_executable):
                return source
    if any(source.backend == "whisper-cpp" for source in allowed) and not cpp_executable:
        raise ValueError("Found a cached whisper.cpp model, but whisper-cli is missing. Install whisper.cpp (macOS: brew install whisper-cpp), or provide --whisper-cpp-bin PATH.")
    if any(source.backend == "mlx" for source in allowed) and not use_mlx:
        raise ValueError("The cached MLX model requires Apple Silicon and mlx-whisper (run uv sync)")
    if offline:
        raise ValueError(f"No compatible cached {name!r} model found for {backend}; --offline prevents downloading")
    if backend in ("macwhisper", "whisper-cpp"):
        raise ValueError(f"No installed {name!r} model found for {backend}. Select a cached model with --model or --model-dir.")
    engine = ("mlx" if use_mlx and name in MODEL_NAMES else "faster-whisper") if backend == "auto" else backend
    if device not in devices[engine]:
        raise ValueError(f"{engine} does not support --device {device}; select a compatible model/backend")
    if engine in ("whisper", "mlx") and name not in MODEL_NAMES:
        raise ValueError("The whisper backend accepts official model names or local .pt files; use faster-whisper for Hugging Face repositories")
    return ModelSource(engine, faster_repo(name) if engine == "faster-whisper" else name, name, "download", cached=False)
