"""Create a local environment, install dependencies, and optionally train."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
TORCH_INDEX = "https://download.pytorch.org/whl"


CUDA_VERSION_RE = re.compile(r"CUDA\s+Version\s*:\s*(\d+)\.(\d+)", re.IGNORECASE)


def nvidia_smi_paths():
    """PATH에 없더라도 OS별 기본 설치 위치의 nvidia-smi를 찾는다."""
    candidates = [shutil.which("nvidia-smi")]
    if sys.platform == "win32":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates += [Path(system_root) / "System32" / "nvidia-smi.exe",
                       Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"]
    else:
        candidates += ["/usr/bin/nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi",
                       "/usr/local/nvidia/bin/nvidia-smi"]
    paths = []
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            resolved = str(Path(candidate).resolve())
            if resolved not in paths:
                paths.append(resolved)
    return paths


def cuda_from_nvidia_smi(notes):
    paths = nvidia_smi_paths()
    if not paths:
        notes.append("nvidia-smi not found on PATH or default locations")
    for executable in paths:
        # 기본 표 형식과 -q 상세 형식을 모두 시도한다.
        for extra in ([], ["-q"]):
            command = [executable, *extra]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                        errors="replace", check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                notes.append(f"{subprocess.list2cmdline(command)}: {exc}")
                continue
            # 경고로 종료 코드가 0이 아니어도 버전이 출력됐다면 사용한다.
            match = CUDA_VERSION_RE.search(result.stdout + result.stderr)
            if match:
                if result.returncode != 0:
                    print(f"nvidia-smi exited with {result.returncode}, "
                          "but reported a CUDA version; using it.", flush=True)
                return tuple(map(int, match.groups()))
            output = (result.stdout + result.stderr).strip().splitlines()
            notes.append(f"{subprocess.list2cmdline(command)} (exit {result.returncode}): "
                         f"{output[0] if output else 'no output'}")
    return None


def cuda_from_driver_library(notes):
    """nvidia-smi 없이 드라이버 라이브러리의 cuDriverGetVersion으로 확인한다."""
    if sys.platform == "win32":
        names = ["nvcuda.dll"]
        loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
    else:
        names = ["libcuda.so.1", "libcuda.so", "/usr/lib/wsl/lib/libcuda.so.1"]
        loader = ctypes.CDLL
    for name in names:
        try:
            library = loader(name)
        except OSError:
            continue
        version = ctypes.c_int()
        status = library.cuDriverGetVersion(ctypes.byref(version))
        if status == 0 and version.value > 0:
            return version.value // 1000, (version.value % 1000) // 10
        notes.append(f"{name}: cuDriverGetVersion returned {status}")
    notes.append(f"CUDA driver library not loadable ({', '.join(names)})")
    return None


def detect_cuda_limit():
    """드라이버가 지원하는 최대 CUDA 버전을 반환한다. nvidia-smi → 드라이버 라이브러리 순."""
    notes = []
    limit = cuda_from_nvidia_smi(notes) or cuda_from_driver_library(notes)
    if limit is None:
        print("CUDA detection details:", *(f"  - {note}" for note in notes),
              sep="\n", flush=True)
    return limit


def cuda_tags_for_driver():
    """드라이버가 지원하는 공식 CUDA 인덱스 태그를 최신순으로 반환한다."""
    limit = detect_cuda_limit()
    if limit is None:
        return None
    print(f"Driver supports CUDA up to {limit[0]}.{limit[1]}", flush=True)
    with urlopen(f"{TORCH_INDEX}/", timeout=30) as response:
        listing = response.read().decode("utf-8")
    tags = set(re.findall(r'href=[\"\'](?:[^\"\']*/)?(cu\d{2,})/?[\"\']', listing))

    def version(tag):
        digits = tag[2:]
        return int(digits[:-1]), int(digits[-1])
    tags = [tag for tag in sorted(tags, key=version, reverse=True) if version(tag) <= limit]
    if not tags:
        raise RuntimeError("No official CUDA index matches the driver. Update the driver "
                           "or explicitly select --torch-backend cpu.")
    return tags


def torch_candidates(backend):
    """(태그, 인덱스 URL, 휠 파일명의 로컬 버전 태그) 후보를 우선순위 순으로 반환한다."""
    if backend == "cpu":
        if sys.platform == "darwin":
            return [("cpu", "https://pypi.org/simple", None)]
        return [("cpu", f"{TORCH_INDEX}/cpu", "cpu")]
    if backend != "auto":
        return [(backend, f"{TORCH_INDEX}/{backend}", backend)]
    tags = cuda_tags_for_driver() if sys.platform != "darwin" else None
    if tags is None:
        print("NVIDIA CUDA driver not detected; selecting CPU/default platform wheels. "
              "If a GPU is installed, pass --torch-backend cuXXX (e.g. cu130).", flush=True)
        return torch_candidates("cpu")
    return [(tag, f"{TORCH_INDEX}/{tag}", tag) for tag in tags]


# ------------------------------------------------------------ platform / wheels
def version_key(text):
    return tuple(int(part) for part in re.findall(r"\d+", text))


def platform_supports(platform_tag):
    """휠의 플랫폼 태그를 이 OS·아키텍처에서 설치할 수 있는지 판단한다."""
    machine = platform.machine().lower()
    if sys.platform == "win32":
        expected = {"amd64": "win_amd64", "x86_64": "win_amd64", "arm64": "win_arm64"}
        return platform_tag == expected.get(machine)
    if sys.platform.startswith("linux"):
        arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
        legacy = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}
        match = re.fullmatch(rf"manylinux_(\d+)_(\d+)_{arch}", platform_tag)
        if match:
            required = tuple(map(int, match.groups()))
        else:
            name, _, tag_arch = platform_tag.partition("_")
            if name not in legacy or tag_arch != arch:
                return False
            required = legacy[name]
        libc, glibc = platform.libc_ver()
        return libc == "glibc" and version_key(glibc)[:2] >= required
    if sys.platform == "darwin":
        match = re.fullmatch(r"macosx_(\d+)_(\d+)_(\w+)", platform_tag)
        if not match:
            return False
        arch = "arm64" if machine == "arm64" else "x86_64"
        current = version_key(platform.mac_ver()[0])[:2]
        return (match.group(3) in (arch, "universal2")
                and current >= (int(match.group(1)), int(match.group(2))))
    return False


WHEEL_RE = re.compile(
    r"(?P<name>torch|torchvision)-(?P<version>\d+(?:\.\d+)*)(?:\+(?P<local>[\w.]+))?"
    r"-(?P<python>cp\d+)-(?P<abi>cp\d+t?)-(?P<platform>[\w.]+)\.whl")


def available_wheels(index, local):
    """{패키지: {abi: [버전...]}} — 이 플랫폼에 설치 가능한 휠만 모은다."""
    wheels = {}
    for package in ("torch", "torchvision"):
        with urlopen(f"{index}/{package}/", timeout=60) as response:
            listing = response.read().decode("utf-8", errors="replace")
        found = {}
        for match in WHEEL_RE.finditer(listing):
            if (match["name"] != package or match["local"] != local
                    or not any(platform_supports(tag) for tag in match["platform"].split("."))):
                continue
            found.setdefault(match["abi"], set()).add(match["version"])
        wheels[package] = found
    return wheels


def torch_pair(wheels, abi):
    """해당 ABI에서 서로 맞는 최신 (torch, torchvision) 버전 쌍.

    torchvision 0.N은 torch 2.(N-15)와 짝을 이룬다 (0.15 <-> 2.0부터 유지되는 규칙).
    """
    torch_versions = wheels["torch"].get(abi, set())
    for tv in sorted(wheels["torchvision"].get(abi, set()), key=version_key, reverse=True):
        major, minor = version_key(tv)[:2]
        if major != 0 or minor < 15:
            continue
        matches = [t for t in torch_versions if version_key(t)[:2] == (2, minor - 15)]
        if matches:
            return max(matches, key=version_key), tv
    return None


# ------------------------------------------------------------ Python interpreters
PROBE = ("import json, struct, sys, sysconfig, importlib.util as u; "
         "print(json.dumps({'version': list(sys.version_info[:2]), "
         "'abi': 'cp%d%d' % sys.version_info[:2] + ('t' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''), "
         "'bits': struct.calcsize('P') * 8, "
         "'venv': bool(u.find_spec('venv') and u.find_spec('ensurepip')), "
         "'executable': sys.executable}))")


def probe_python(executable):
    try:
        result = subprocess.run([str(executable), "-I", "-c", PROBE], capture_output=True,
                                text=True, timeout=30, check=False)
        info = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if info:
        info["executable"] = str(Path(info["executable"]).resolve())
    return info


def find_pythons(preferred=None):
    """설치된 CPython 인터프리터를 찾는다. 첫 항목이 선호 인터프리터."""
    candidates = [preferred or sys.executable]
    if sys.platform == "win32" and shutil.which("py"):
        try:
            listing = subprocess.run(["py", "-0p"], capture_output=True, text=True,
                                     timeout=30, check=False).stdout
            candidates += re.findall(r"([A-Za-z]:\\.*?python(?:\d+(?:\.\d+)?)?t?\.exe)", listing,
                                     flags=re.IGNORECASE)
        except (OSError, subprocess.TimeoutExpired):
            pass
    names = ["python3", "python"] + [f"python3.{minor}{suffix}" for minor in range(8, 20)
                                     for suffix in ("", "t")]
    candidates += [shutil.which(name) for name in names]
    pythons, seen = [], set()
    for candidate in candidates:
        if not candidate:
            continue
        info = probe_python(candidate)
        if info and info["executable"] not in seen and info["bits"] == 64:
            seen.add(info["executable"])
            pythons.append(info)
    return pythons


def describe(info):
    return f"Python {info['version'][0]}.{info['version'][1]} ({info['abi']}) {info['executable']}"


def plan_install(backend, venv_python, python_choice, recreate):
    """CUDA 태그·Python·torch/torchvision 버전을 함께 고른다.

    우선순위: 최신 CUDA 태그 > 지정/기존/현재 Python > 높은 Python 버전(GIL 빌드 우선).
    기존 .venv는 --recreate-venv 없이는 그 Python으로 고정한다.
    """
    keep_venv = venv_python.is_file() and not recreate
    if keep_venv:
        info = probe_python(venv_python)
        if info is None:
            raise RuntimeError(f"Cannot run the existing environment's Python: {venv_python}")
        pythons = [info]
        print(f"Existing environment: {describe(info)}", flush=True)
    else:
        preferred = None
        if python_choice:
            preferred = shutil.which(python_choice) or python_choice
            if re.fullmatch(r"\d+\.\d+t?", python_choice):
                preferred = shutil.which(f"python{python_choice}") or preferred
                if sys.platform == "win32" and shutil.which("py"):
                    result = subprocess.run(["py", f"-{python_choice}", "-c",
                                             "import sys; print(sys.executable)"],
                                            capture_output=True, text=True, check=False)
                    preferred = result.stdout.strip() or preferred
            if probe_python(preferred) is None:
                raise RuntimeError(f"--python {python_choice} is not a runnable Python")
        pythons = find_pythons(preferred)
        if python_choice:
            pythons = pythons[:1]
        print("Python interpreters found:", *(f"  - {describe(i)}" for i in pythons),
              sep="\n", flush=True)

    supported_by_tag = {}
    for tag, index, local in torch_candidates(backend):
        print(f"Checking {tag} wheels for this platform...", flush=True)
        wheels = available_wheels(index, local)
        compatible = []
        for rank, info in enumerate(pythons):
            pair = torch_pair(wheels, info["abi"])
            if pair and (keep_venv or info["venv"]):
                compatible.append((rank, info, pair))
        supported_by_tag[tag] = sorted(
            {abi for abi in wheels["torch"] if torch_pair(wheels, abi)}, key=version_key)
        if not compatible:
            print(f"No usable torch/torchvision pair in {tag} for the Python(s) above.", flush=True)
            continue
        # 선호 인터프리터(목록 첫 항목)가 맞으면 그대로, 아니면 높은 버전(GIL 빌드 우선)을 고른다.
        rank, info, pair = min(compatible, key=lambda item: (
            item[0] != 0, item[1]["abi"].endswith("t"), [-n for n in item[1]["version"]]))
        return {"tag": tag, "index": index, "python": info, "torch": pair[0], "torchvision": pair[1]}

    hints = "; ".join(f"{tag}: {', '.join(abis) or 'none'}" for tag, abis in supported_by_tag.items())
    if keep_venv:
        raise RuntimeError(
            "The existing .venv Python has no compatible torch/torchvision wheels "
            f"(supported per index -> {hints}). Rerun with --recreate-venv "
            "(optionally --python 3.12) to rebuild it with a compatible Python.")
    raise RuntimeError(
        f"No installed Python matches the available wheels (supported -> {hints}). "
        "Install one of those Python versions (with venv/ensurepip), or use --torch-backend cpu.")


def install_dependencies(python, requirements, plan):
    tag, index = plan["tag"], plan["index"]
    pins = {"torch": plan["torch"], "torchvision": plan["torchvision"]}
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"],
                   cwd=ROOT, check=True)
    print(f"Selected PyTorch backend: {tag} ({index}) "
          f"torch {pins['torch']}, torchvision {pins['torchvision']}", flush=True)
    # 재사용 환경에 CPU/CUDA의 다른 빌드가 남지 않도록 두 휠을 교체한다.
    # 하위 의존성은 아래 requirements 설치에서 함께 해결한다.
    local = f"+{tag}" if "pytorch.org" in index else ""
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "--force-reinstall",
         "--no-deps", "--only-binary", ":all:",
         *(f"{name}=={version}{local}" for name, version in pins.items()),
         "--index-url", index], cwd=ROOT, check=True,
    )
    # 나머지 패키지를 갱신할 때 선택한 빌드가 다른 빌드로 바뀌지 않게 한다.
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
        "--python",
        help="Interpreter for a new .venv: a version such as 3.12 or a path. "
             "Default: this Python if compatible, otherwise the newest compatible one found.",
    )
    parser.add_argument(
        "--recreate-venv", action="store_true",
        help="Delete and rebuild .venv (needed to change the Python of an existing environment).",
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
        plan = plan_install(args.torch_backend, python, args.python, args.recreate_venv)
        base = plan["python"]
        print(f"Selected {describe(base)}", flush=True)
        if args.recreate_venv and env_dir.exists():
            print(f"Removing environment: {env_dir}", flush=True)
            shutil.rmtree(env_dir)
        if not env_dir.exists():
            print(f"Creating environment: {env_dir}", flush=True)
            subprocess.run([base["executable"], "-m", "venv", str(env_dir)], cwd=ROOT, check=True)
        if not (env_dir / "pyvenv.cfg").is_file() or not python.is_file():
            raise RuntimeError(f"Incomplete virtual environment: {env_dir}")

        print(f"Installing dependencies with {python}", flush=True)
        install_dependencies(python, requirements, plan)
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
