"""Create a local environment, install dependencies, and optionally train."""

import argparse
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent


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
        "training_args", nargs=argparse.REMAINDER,
        help="Arguments passed to training after --.",
    )
    args = parser.parse_args(argv)
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
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade",
             "--upgrade-strategy", "eager", "-r", str(requirements)],
            cwd=ROOT, check=True,
        )
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
    except (OSError, RuntimeError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
