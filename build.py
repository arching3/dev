"""Create a local environment, install dependencies, and optionally train."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
import venv


ROOT = Path(__file__).resolve().parent
TORCH_INDEX = "https://download.pytorch.org/whl"


def detect_cuda_limit():
    """nvidia-smi가 보고하는 드라이버의 최대 지원 CUDA 버전을 반환한다."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15,
            errors="replace", check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if result.returncode != 0 or match is None:
        return None
    return tuple(map(int, match.groups()))


def torch_candidates(backend):
    """CUDA 버전을 하드코딩하지 않고 공식 인덱스에서 후보를 얻는다."""
    if backend == "cpu":
        return [("cpu", "https://pypi.org/simple" if sys.platform == "darwin"
                 else f"{TORCH_INDEX}/cpu")]
    if backend != "auto":
        return [(backend, f"{TORCH_INDEX}/{backend}")]
    limit = detect_cuda_limit() if sys.platform != "darwin" else None
    if limit is None:
        print("NVIDIA CUDA driver not detected; selecting CPU/default platform wheels.", flush=True)
        return torch_candidates("cpu")

    print(f"Driver supports CUDA up to {limit[0]}.{limit[1]}", flush=True)
    with urlopen(f"{TORCH_INDEX}/", timeout=30) as response:
        listing = response.read().decode("utf-8")
    tags = set(re.findall(r'href=[\"\'](?:[^\"\']*/)?(cu\d{2,})/?[\"\']', listing))
    def version(tag):
        digits = tag[2:]
        return int(digits[:-1]), int(digits[-1])
    candidates = [(tag, f"{TORCH_INDEX}/{tag}")
                  for tag in sorted(tags, key=version, reverse=True)
                  if version(tag) <= limit]
    if not candidates:
        raise RuntimeError("No official CUDA index matches the driver. Update the driver "
                           "or explicitly select --torch-backend cpu.")
    return candidates


def select_torch_index(python, backend):
    """가상환경 pip가 Python ABI/OS/아키텍처에 맞는 휠 존재 여부를 판단한다."""
    for tag, index in torch_candidates(backend):
        print(f"Checking {tag} wheels for this Python/platform...", flush=True)
        for package in ("torch", "torchvision"):
            result = subprocess.run(
                [str(python), "-m", "pip", "--disable-pip-version-check", "index",
                 "versions", package, "--index-url", index, "--only-binary", ":all:"],
                capture_output=True, text=True, errors="replace", check=False,
            )
            if result.returncode != 0:
                details = result.stdout + result.stderr
                if "No matching distribution found" not in details:
                    raise RuntimeError(f"Cannot query {index}:\n{details}")
                print(f"No compatible {package} wheel in {tag}; trying the next candidate.", flush=True)
                break
        else:
            return tag, index
    raise RuntimeError("No compatible torch/torchvision wheels for this Python/platform "
                       "and driver. Use a supported Python version, update the driver, "
                       "or explicitly select --torch-backend cpu.")


def install_dependencies(python, requirements, backend):
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"],
                   cwd=ROOT, check=True)
    tag, index = select_torch_index(python, backend)
    print(f"Selected PyTorch backend: {tag} ({index})", flush=True)
    # 재사용 환경에 CPU/CUDA의 다른 빌드가 남지 않도록 두 휠을 교체한다.
    # 하위 의존성은 아래 requirements 설치에서 함께 해결한다.
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "--force-reinstall",
         "--no-deps", "--only-binary", ":all:", "torch", "torchvision",
         "--index-url", index], cwd=ROOT, check=True,
    )
    # 나머지 패키지를 갱신할 때 선택한 CUDA 빌드가 PyPI 빌드로 바뀌지 않게 한다.
    versions = subprocess.run(
        [str(python), "-c", "import importlib.metadata as m, json; "
         "print(json.dumps({p: m.version(p) for p in ('torch', 'torchvision')}))"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    with tempfile.TemporaryDirectory(prefix="week10-torch-") as directory:
        constraints = Path(directory) / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version}\n"
                                       for name, version in json.loads(versions.stdout).items()),
                               encoding="utf-8")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "--upgrade-strategy",
             "eager", "-r", str(requirements), "-c", str(constraints),
             "--extra-index-url", index], cwd=ROOT, check=True,
        )
    subprocess.run([str(python), "-m", "pip", "check"], cwd=ROOT, check=True)
    subprocess.run(
        [str(python), "-c", "import torch; "
         "print('PyTorch:', torch.__version__); "
         "print('CUDA runtime:', torch.version.cuda); "
         "print('cuDNN:', torch.backends.cudnn.version()); "
         "print('CUDA available:', torch.cuda.is_available())"], cwd=ROOT, check=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup-only", action="store_true",
        help="Install the environment and exit without running training.",
    )
    parser.add_argument(
        "--script", default="train.py",
        help="Training script relative to this directory (default: train.py).",
    )
    parser.add_argument(
        "--torch-backend", default="auto",
        help="auto (driver + Python wheel detection), cpu, or a CUDA tag such as cu128.",
    )
    parser.add_argument(
        "training_args", nargs=argparse.REMAINDER,
        help="Arguments passed to training after --.",
    )
    args = parser.parse_args(argv)
    if args.torch_backend not in ("auto", "cpu") and not re.fullmatch(r"cu\d{2,}", args.torch_backend):
        parser.error("--torch-backend must be auto, cpu, or a CUDA tag such as cu128")
    training_args = args.training_args
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    if args.setup_only and training_args:
        parser.error("--setup-only cannot be combined with training arguments")

    requirements = ROOT / "requirements.txt"
    script = ROOT / args.script
    if not requirements.is_file():
        parser.error(f"Requirements file not found: {requirements}")
    if not args.setup_only and not script.is_file():
        parser.error(
            f"Training script not found: {script}. "
            "Add train.py, select --script FILE, or use --setup-only for notebooks."
        )

    env_dir = ROOT / ".venv"
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    try:
        if not env_dir.exists():
            print(f"Creating environment: {env_dir}", flush=True)
            venv.EnvBuilder(with_pip=True).create(env_dir)
        if not (env_dir / "pyvenv.cfg").is_file() or not python.is_file():
            raise RuntimeError(f"Incomplete virtual environment: {env_dir}")

        print(f"Installing dependencies with {python}", flush=True)
        install_dependencies(python, requirements, args.torch_backend)
        if args.setup_only:
            print(f"Environment ready: {env_dir}")
            print(f"Notebook interpreter: {python}")
            print("Select this interpreter in your editor, or launch Jupyter with:")
            print(subprocess.list2cmdline([str(python), "-m", "jupyterlab"]))
            print("JupyterLab uses this environment's Python 3 (ipykernel) kernel.")
            return 0

        print(f"Running training: {script}", flush=True)
        subprocess.run(
            [str(python), str(script), *training_args], cwd=ROOT, check=True,
        )
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Command failed (exit {exc.returncode}): {exc.cmd}", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 128 - exc.returncode
    except (OSError, RuntimeError, URLError, ValueError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
