"""현재 브랜치를 GitHub과 동기화하면서 train.py를 실행하는 비동기 TUI.

사용법:
  python build.py --setup-only      # .venv 구성 (textual 포함)
  .venv/bin/python watchdog.py      # Windows: .venv\\Scripts\\python.exe watchdog.py

동작:
  - 일정 간격으로 `git fetch` 후 원격 브랜치와 HEAD를 비교한다.
  - 원격에 새 커밋이 있으면 `git reset --hard <원격>`으로 즉시 반영한다.
    로컬에서 수정한 추적 파일과 로컬 커밋은 경고 없이 덮어쓴다.
    (추적되지 않는 파일 — ckpts/, histories/ 등 학습 결과 — 는 지우지 않는다.)
  - 학습 중에 새 커밋이 오면 화면 안의 질문 패널로 처리 방법을 묻는다.
  - 이 파일(watchdog.py)은 실행 중 바뀌지 않는다고 가정한다.
  - 노트북(.ipynb)을 셀 단위로 표시하고 선택한 Python의 Jupyter 커널로 실행한다.
    기본 커널은 프로젝트와 상위 폴더의 .venv에서 찾으며, 없으면 현재 Python을 쓴다.
    커널 경로에 Python 실행 파일 또는 가상환경 폴더를 입력한 뒤 적용할 수 있다.
    셀을 수정·생성하지 않으며 실행 결과는 화면에만 표시한다(파일에 쓰지 않음).
    이미지 출력은 .watchdog/<노트북>/ 아래 PNG로 저장하고 경로를 보여준다.
    원격 변경으로 노트북이 바뀌면 다시 불러오고 바뀐 셀을 표시한다.
    셀 실행 중 원격 변경이 오면 알린 뒤 3초 후 실행을 중단하고 반영한다.

단축키: ←/→ 셀 이동 · F7 셀 실행 후 다음 셀 · F8 현재 셀부터 모두 실행 · F9 셀 실행 중단
        F10 커널 재시작 · F2 학습 시작 · F3 학습 중지 · F5 지금 확인 · F6 강제 동기화
        Ctrl+Q 종료
"""

from __future__ import annotations

import asyncio
import base64
import codecs
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import sys

try:
    from rich.console import Group
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import (Button, ContentSwitcher, Footer, Input, Label, RichLog, Select,
                                 SelectionList, Static)
    from textual.widgets.selection_list import Selection
except ImportError:  # pragma: no cover
    sys.exit("textual이 필요합니다. `python build.py --setup-only`로 환경을 구성한 뒤 "
             ".venv의 Python으로 실행하거나 `pip install textual`을 실행하세요.")

try:  # 노트북 실행은 jupyter_client가 있을 때만 사용한다(requirements.txt의 jupyter에 포함).
    from jupyter_client.manager import AsyncKernelManager
except ImportError:  # pragma: no cover
    AsyncKernelManager = None


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = "train.py"
DEFAULT_INTERVAL = 30  # 초
IS_WINDOWS = sys.platform == "win32"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
LIST_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+)$")
OUTPUT_DIR = ROOT / ".watchdog"          # 노트북 이미지 출력 (.gitignore에 포함)
SKIP_DIRS = {".venv", "venv", ".git", ".ipynb_checkpoints", ".watchdog"}
PREEMPT_DELAY = 3                         # 셀 실행 중 원격 변경 시 중단까지 대기(초)


def now():
    return datetime.now().strftime("%H:%M:%S")


def venv_python():
    """build.py가 만든 .venv의 Python을 우선 사용하고, 없으면 현재 인터프리터를 쓴다."""
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    return str(candidate) if candidate.is_file() else sys.executable


def notebook_python():
    """프로젝트에서 가까운 .venv를 찾아 노트북의 기본 Python으로 선택한다."""
    for directory in (ROOT, *ROOT.parents):
        candidate = directory / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def kernel_python_path(value):
    """실행 파일 또는 가상환경 경로를 검증한다. 상대 경로는 ROOT 기준이다."""
    if not value.strip():
        raise ValueError("Python 실행 파일 또는 가상환경 폴더 경로를 입력하세요.")
    path = Path(os.path.expandvars(value.strip())).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        path = path / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    # resolve()는 venv의 Python 심볼릭 링크를 시스템 Python으로 바꾸므로 쓰지 않는다.
    path = Path(os.path.abspath(path))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"실행 가능한 Python 파일이 없습니다: {path}")
    return str(path)


def child_env():
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", COLUMNS="110")
    # 인증 프롬프트가 TUI를 막지 않도록 비대화식으로 git을 실행한다.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    return env


async def run(*cmd, timeout=120):
    """명령을 비동기로 실행하고 (종료 코드, stdout, stderr)를 반환한다."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=ROOT, env=child_env(), stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"timeout after {timeout}s: {' '.join(cmd)}"
    return (proc.returncode, out.decode("utf-8", "replace").strip(),
            err.decode("utf-8", "replace").strip())


async def git(*args, timeout=120):
    return await run("git", "-c", "core.quotepath=off", *args, timeout=timeout)


class GitError(RuntimeError):
    pass


async def git_ok(*args, timeout=120):
    code, out, err = await git(*args, timeout=timeout)
    if code != 0:
        raise GitError(err or out or f"git {' '.join(args)} failed ({code})")
    return out


# ------------------------------------------------------------------ notebooks
def find_notebooks():
    found = []
    for path in sorted(ROOT.rglob("*.ipynb")):
        rel = path.relative_to(ROOT)
        if not SKIP_DIRS.intersection(rel.parts):
            found.append(rel.as_posix())
    return found


def load_cells(rel_path):
    """노트북의 셀을 [{key, type, source}] 로 읽는다. key는 셀 id(없으면 순번)."""
    data = json.loads((ROOT / rel_path).read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(data.get("cells", [])):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        cells.append({"key": cell.get("id") or f"#{index}",
                      "type": cell.get("cell_type", "raw"), "source": source})
    return cells


def collapse_cr(text):
    """'\\r'로 덮어쓰는 진행률 출력(tqdm 등)을 마지막 상태만 남긴다."""
    lines = []
    for line in text.split("\n"):
        parts = [part for part in line.split("\r") if part]
        lines.append(parts[-1] if parts else "")
    return "\n".join(lines)


class CellRun:
    """셀 하나의 실행 결과(화면 표시 전용)."""

    def __init__(self):
        self.status = "실행 중"
        self.count = None
        self.items = []  # ["stream", name, text] 또는 ["rich", renderable]

    def add_stream(self, name, text):
        if self.items and self.items[-1][0] == "stream" and self.items[-1][1] == name:
            self.items[-1][2] += text
        else:
            self.items.append(["stream", name, text])

    def renderables(self):
        for item in self.items:
            if item[0] == "stream":
                text = Text.from_ansi(collapse_cr(item[2]).rstrip("\n"))
                if item[1] == "stderr":
                    text.stylize("red")
                yield text
            else:
                yield item[1]


class Watchdog(App):
    TITLE = "watchdog"
    CSS = """
    Screen { layout: vertical; }
    #status { height: auto; padding: 0 1; background: $boost; }
    #prompt { display: none; height: auto; padding: 0 1; border: heavy $warning; }
    #prompt.visible { display: block; }
    #prompt Horizontal { height: auto; }
    #prompt Button { margin-right: 1; }
    #body { height: 1fr; }
    #left { width: 44; padding: 0 1; }
    #left Label { margin-top: 1; color: $text-muted; }
    #experiments { height: 1fr; min-height: 5; }
    #left Horizontal { height: auto; margin-top: 1; }
    #left Button { width: 1fr; min-width: 8; }
    #right { width: 1fr; }
    #notebook { height: 3fr; border: round $success; }
    #nb_bar { height: auto; }
    #nb_select { width: 1fr; }
    #nb_bar Button { margin-left: 1; }
    #kernel_bar { height: auto; }
    #kernel_bar Label { width: auto; padding: 1; }
    #kernel_path { width: 1fr; }
    #kernel_bar Button { margin-left: 1; }
    #nb_view { height: 1fr; padding: 0 1; }
    #output { height: 2fr; border: round $primary; }
    #out_switch, #train_log, #cell_out { height: 1fr; }
    #cell_out { padding: 0 1; }
    #progress { height: 1; padding: 0 1; color: $accent; }
    #git_log { height: 1fr; border: round $secondary; }
    """
    BINDINGS = [
        ("f2", "train", "학습 시작"),
        ("f4", "toggle_output", "출력 전환"),
        ("f3", "stop", "학습 중지"),
        ("f5", "check", "지금 확인"),
        ("f6", "force_sync", "강제 동기화"),
        ("left", "move_cell(-1)", "이전 셀"),
        ("right", "move_cell(1)", "다음 셀"),
        ("f7", "run_cell", "셀 실행"),
        ("f8", "run_all", "모두 실행"),
        ("f9", "interrupt_cells", "실행 중단"),
        ("f10", "restart_kernel", "커널 재시작"),
        ("ctrl+q", "quit", "종료"),
    ]

    def __init__(self):
        super().__init__()
        self.python = venv_python()
        self.kernel_python = notebook_python()
        self.interval = DEFAULT_INTERVAL
        self.branch = "?"
        self.remote = "origin"
        self.remote_branch = "?"
        self.local_sha = ""
        self.remote_sha = ""
        self.dirty = 0
        self.last_check = "-"
        self.check_error = ""
        self.git_lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.proc: asyncio.subprocess.Process | None = None
        self.train_args: list[str] = []
        self.train_started = ""
        self.prompt_sha = ""      # 질문 패널에 표시 중인 원격 커밋
        self.deferred = False     # "학습 끝나면 반영"을 선택함
        # 노트북
        self.nb_path = ""         # ROOT 기준 상대 경로
        self.cells: list[dict] = []
        self.cell_pos = 0
        self.changed_keys: set[str] = set()   # 마지막 동기화에서 바뀐 셀
        self.runs: dict[str, CellRun] = {}    # 셀 key → 실행 결과
        self.running_key = ""
        self.km = None
        self.kc = None
        self.kernel_lock = asyncio.Lock()
        self.exec_task: asyncio.Task | None = None
        self.stop_queue = False

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Static(id="status")
        with Vertical(id="prompt"):
            yield Label(id="prompt_text")
            with Horizontal():
                yield Button("중단 → 반영 → 재시작", id="ans_restart", variant="warning")
                yield Button("중단 → 반영", id="ans_stop", variant="error")
                yield Button("학습 끝나면 반영", id="ans_defer", variant="primary")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Label("실험 선택 (Space로 토글)")
                yield SelectionList[str](id="experiments")
                yield Label("추가 인자 (예: --epochs 5 --lr 0.01)")
                yield Input(placeholder="train.py 인자", id="extra_args")
                yield Label("확인 간격(초)")
                yield Input(str(self.interval), type="integer", id="interval")
                with Horizontal():
                    yield Button("학습", id="btn_train", variant="success")
                    yield Button("중지", id="btn_stop", variant="error")
                with Horizontal():
                    yield Button("확인", id="btn_check")
                    yield Button("동기화", id="btn_sync", variant="warning")
            with Vertical(id="right"):
                with Vertical(id="notebook"):
                    with Horizontal(id="nb_bar"):
                        yield Select([], prompt="노트북 선택", id="nb_select", compact=True)
                        yield Button("▶ 셀", id="nb_run", compact=True)
                        yield Button("▶▶ 모두", id="nb_runall", compact=True)
                        yield Button("■ 중단", id="nb_interrupt", compact=True)
                        yield Button("↻ 커널", id="nb_restart", compact=True)
                    with Horizontal(id="kernel_bar"):
                        yield Label("커널 Python")
                        yield Input(self.kernel_python, placeholder="Python 실행 파일 / 가상환경 폴더",
                                    id="kernel_path")
                        yield Button("적용", id="kernel_apply", compact=True)
                        yield Button(".venv 자동", id="kernel_auto", compact=True)
                    with VerticalScroll(id="nb_view"):
                        yield Static(id="nb_cell")
                with Vertical(id="output"):
                    with ContentSwitcher(initial="train_log", id="out_switch"):
                        yield RichLog(id="train_log", wrap=True, max_lines=5000, markup=False)
                        with VerticalScroll(id="cell_out"):
                            yield Static(id="cell_out_text")
                yield Static(id="progress")
                yield RichLog(id="git_log", wrap=True, max_lines=1000, markup=False)
        yield Footer()

    async def on_mount(self):
        self.query_one("#git_log", RichLog).border_title = "git / watchdog"
        self.query_one("#experiments", SelectionList).border_title = "experiments"
        self.set_interval(1, self.render_status)
        self.show_output("train")
        self.refresh_notebook_list()
        self.render_cell()
        self.log_git(f"Python: {self.python}")
        self.log_git(f"노트북 커널 Python: {self.kernel_python}")
        try:
            await self.resolve_branch()
        except GitError as exc:
            self.log_git(f"브랜치 확인 실패: {exc}", "red")
            return
        self.run_worker(self.watch_loop(), name="watch", group="watch")
        self.run_worker(self.load_experiments(), group="experiments", exclusive=True)

    # ----------------------------------------------------------------- helpers
    def log_git(self, message, style=""):
        self.query_one("#git_log", RichLog).write(Text(f"[{now()}] {message}", style=style))

    def log_train(self, line, style=""):
        self.query_one("#train_log", RichLog).write(Text(line, style=style))

    @property
    def training(self):
        return self.proc is not None and self.proc.returncode is None

    def render_status(self):
        self.update_output_title()
        short = lambda sha: sha[:7] if sha else "-------"
        sync = ("[green]최신[/]" if self.local_sha and self.local_sha == self.remote_sha
                else "[yellow]변경 대기[/]" if self.remote_sha else "[dim]?[/]")
        if self.training:
            train = f"[green]학습 중[/] ({self.train_started}~) {' '.join(self.train_args)}"
        else:
            train = "[dim]대기[/]"
        dirty = f" · [yellow]로컬 수정 {self.dirty}개[/]" if self.dirty else ""
        error = f"\n[red]{self.check_error}[/]" if self.check_error else ""
        deferred = " · [cyan]학습 후 반영 예약[/]" if self.deferred else ""
        self.query_one("#status", Static).update(
            f"[b]{self.branch}[/] ← {self.remote}/{self.remote_branch} · "
            f"local {short(self.local_sha)} / remote {short(self.remote_sha)} {sync}{dirty} · "
            f"확인 {self.last_check} ({self.interval}s){deferred}\n학습: {train}{error}")

    # -------------------------------------------------------------------- git
    async def resolve_branch(self):
        self.branch = await git_ok("rev-parse", "--abbrev-ref", "HEAD")
        if self.branch == "HEAD":
            raise GitError("detached HEAD 상태입니다. 브랜치로 전환한 뒤 실행하세요.")
        code, upstream, _ = await git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if code == 0 and "/" in upstream:
            self.remote, self.remote_branch = upstream.split("/", 1)
        else:
            self.remote, self.remote_branch = "origin", self.branch
        self.local_sha = await git_ok("rev-parse", "HEAD")
        self.log_git(f"브랜치 {self.branch} ↔ {self.remote}/{self.remote_branch} 감시 시작")

    @property
    def remote_ref(self):
        return f"refs/remotes/{self.remote}/{self.remote_branch}"

    async def watch_loop(self):
        first = True
        while True:
            await self.check_remote(force=first)
            first = False
            try:
                await asyncio.wait_for(self.wake.wait(), self.interval)
            except TimeoutError:
                pass
            self.wake.clear()

    async def check_remote(self, force=False):
        """원격을 가져와 비교하고, 필요하면 반영하거나 사용자에게 묻는다."""
        try:
            async with self.git_lock:
                await git_ok("fetch", "--quiet", self.remote,
                             f"+refs/heads/{self.remote_branch}:{self.remote_ref}")
                self.remote_sha = await git_ok("rev-parse", self.remote_ref)
                self.local_sha = await git_ok("rev-parse", "HEAD")
                status = await git_ok("status", "--porcelain", "--untracked-files=no")
                self.dirty = len(status.splitlines())
            self.check_error = ""
        except GitError as exc:
            self.check_error = f"원격 확인 실패: {exc.args[0].splitlines()[-1]}"
            self.log_git(self.check_error, "red")
            return
        finally:
            self.last_check = now()

        changed = self.remote_sha != self.local_sha
        if not changed and not (force and self.dirty):
            return
        if not self.training:
            await self.sync(self.remote_sha)
        elif changed:
            await self.ask_during_training(self.remote_sha)

    async def describe(self, sha):
        return await git_ok("log", "-1", "--format=%h %s (%an, %cr)", sha)

    async def sync(self, sha=None):
        """원격 커밋으로 강제 이동한다. 로컬 수정·커밋은 덮어쓰고 추적되지 않는 파일은 둔다."""
        try:
            sha = sha or await git_ok("rev-parse", self.remote_ref)
            clean = not await git_ok("status", "--porcelain", "--untracked-files=no")
            if clean and await git_ok("rev-parse", "HEAD") == sha:
                return False
        except GitError as exc:
            self.log_git(f"동기화 실패: {exc}", "red")
            return False
        await self.preempt_cells()
        try:
            async with self.git_lock:
                sha = sha or await git_ok("rev-parse", self.remote_ref)
                old = await git_ok("rev-parse", "HEAD")
                status = await git_ok("status", "--porcelain", "--untracked-files=no")
                if old == sha and not status:
                    return False
                files = await git_ok("diff", "--name-only", old, sha) if old != sha else ""
                if status:
                    self.log_git(f"로컬 수정 {len(status.splitlines())}개를 덮어씁니다", "yellow")
                await git_ok("reset", "--hard", "--quiet", sha)
                self.local_sha, self.remote_sha, self.dirty = sha, sha, 0
                summary = await self.describe(sha)
        except GitError as exc:
            self.log_git(f"동기화 실패: {exc}", "red")
            return False
        self.log_git(f"반영 완료 → {summary}", "green")
        if files:
            self.log_git("변경 파일: " + ", ".join(files.splitlines()[:20]))
        if "requirements.txt" in files.splitlines():
            self.log_git("requirements.txt가 바뀌었습니다. 종료 후 "
                         "`python build.py --setup-only`로 패키지를 갱신하세요.", "yellow")
        self.hide_prompt()
        self.run_worker(self.load_experiments(), group="experiments", exclusive=True)
        self.refresh_notebook_list()
        if self.nb_path and (self.nb_path in files.splitlines() or status):
            self.load_notebook(self.nb_path, track_changes=True)
        return True

    # ----------------------------------------------------------------- prompt
    async def ask_during_training(self, sha):
        if self.deferred:
            if sha != self.prompt_sha:
                self.prompt_sha = sha
                self.log_git(f"새 커밋 {sha[:7]} — 학습 종료 후 반영 예정", "cyan")
            return
        if sha == self.prompt_sha and self.query_one("#prompt").has_class("visible"):
            return
        self.prompt_sha = sha
        summary = await self.describe(sha)
        self.log_git(f"학습 중 새 커밋 감지: {summary}", "yellow")
        self.query_one("#prompt_text", Label).update(
            f"[b yellow]학습 중에 원격 변경이 감지되었습니다.[/]  {summary}\n"
            "어떻게 처리할까요? (학습을 계속하면 이미 로드된 코드는 바뀌지 않습니다)")
        self.query_one("#prompt").add_class("visible")
        self.bell()

    def hide_prompt(self):
        self.query_one("#prompt").remove_class("visible")

    async def answer(self, choice):
        self.hide_prompt()
        if choice == "defer":
            self.deferred = True
            self.log_git("학습이 끝나면 반영합니다", "cyan")
            return
        args = list(self.train_args)
        await self.stop_training()
        await self.sync()
        if choice == "restart" and args:
            await self.start_training(args)

    # --------------------------------------------------------------- training
    async def load_experiments(self):
        code, out, err = await run(self.python, TRAIN_SCRIPT, "--list", timeout=300)
        widget = self.query_one("#experiments", SelectionList)
        if code != 0:
            self.log_git(f"실험 목록을 불러오지 못했습니다: {(err or out).splitlines()[-1:]}", "red")
            return
        selected = set(widget.selected)
        widget.clear_options()
        widget.add_options([
            Selection(f"{m[1]}: {m[2]}", m[1], m[1] in selected)
            for m in map(LIST_RE.match, out.splitlines()) if m])

    def build_train_args(self):
        selected = sorted(self.query_one("#experiments", SelectionList).selected, key=int)
        try:
            extra = shlex.split(self.query_one("#extra_args", Input).value, posix=not IS_WINDOWS)
        except ValueError as exc:
            self.notify(f"인자 해석 실패: {exc}", severity="error")
            return None
        args = [*selected, *extra]
        if not args:
            self.notify("실험을 선택하거나 인자를 입력하세요.", severity="warning")
            return None
        return args

    async def start_training(self, args):
        if self.training:
            self.notify("이미 학습 중입니다.", severity="warning")
            return
        kwargs = ({"creationflags": 0x00000200} if IS_WINDOWS  # CREATE_NEW_PROCESS_GROUP
                  else {"start_new_session": True})
        try:
            self.proc = await asyncio.create_subprocess_exec(
                self.python, "-u", TRAIN_SCRIPT, *args, cwd=ROOT, env=child_env(),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, **kwargs)
        except OSError as exc:
            self.log_train(f"학습 실행 실패: {exc}", "red")
            return
        self.train_args, self.train_started = list(args), now()
        self.show_output("train")
        self.log_train(f"[{now()}] $ python {TRAIN_SCRIPT} {' '.join(args)}", "bold cyan")
        self.run_worker(self.pump_output(self.proc), group="train")

    async def pump_output(self, proc):
        """stdout을 읽어 줄은 로그에, tqdm의 '\\r' 갱신은 진행 표시줄에 보낸다."""
        progress = self.query_one("#progress", Static)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buffer = ""
        while chunk := await proc.stdout.read(4096):
            buffer += decoder.decode(chunk)
            parts = re.split(r"(\r\n|\n|\r)", buffer)
            buffer = parts.pop()
            for text, sep in zip(parts[::2], parts[1::2]):
                text = ANSI_RE.sub("", text)
                if sep == "\r":
                    if text.strip():
                        progress.update(Text(text.strip()))
                elif text.strip():
                    self.log_train(text)
        if buffer.strip():
            self.log_train(ANSI_RE.sub("", buffer))
        code = await proc.wait()
        progress.update("")
        style = "green" if code == 0 else "red"
        self.log_train(f"[{now()}] train.py 종료 (exit {code})", f"bold {style}")
        self.notify(f"학습 종료 (exit {code})", severity="information" if code == 0 else "error")
        if self.proc is proc:
            self.proc = None
        self.deferred = False
        self.hide_prompt()
        self.wake.set()  # 미뤄둔 변경이 있으면 바로 반영

    async def stop_training(self):
        proc = self.proc
        if proc is None or proc.returncode is not None:
            return
        self.log_train(f"[{now()}] 학습 중지 요청", "yellow")
        steps = ([signal.CTRL_BREAK_EVENT, None] if IS_WINDOWS
                 else [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
        for sig in steps:
            try:
                if sig is None:
                    proc.kill()
                elif IS_WINDOWS:
                    proc.send_signal(sig)
                else:
                    os.killpg(proc.pid, sig)
            except (ProcessLookupError, OSError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), 10)
                break
            except TimeoutError:
                continue
        # pump_output이 종료 처리를 끝낼 때까지 기다린다.
        while self.proc is proc:
            await asyncio.sleep(0.05)

    # ----------------------------------------------------------------- events
    async def on_button_pressed(self, event: Button.Pressed):
        actions = {
            "btn_train": self.action_train, "btn_stop": self.action_stop,
            "btn_check": self.action_check, "btn_sync": self.action_force_sync,
        }
        if event.button.id in actions:
            await actions[event.button.id]()
        elif event.button.id and event.button.id.startswith("ans_"):
            self.run_worker(self.answer(event.button.id[4:]), group="answer", exclusive=True)
        elif event.button.id in {"kernel_apply", "kernel_auto"}:
            await self.select_kernel(auto=event.button.id == "kernel_auto")
        elif event.button.id and event.button.id.startswith("nb_"):
            await getattr(self, {"nb_run": "action_run_cell", "nb_runall": "action_run_all",
                                 "nb_interrupt": "action_interrupt_cells",
                                 "nb_restart": "action_restart_kernel"}[event.button.id])()

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "interval":
            try:
                self.interval = max(5, int(event.value))
            except ValueError:
                return
            event.input.value = str(self.interval)
            self.log_git(f"확인 간격 {self.interval}초")
            self.wake.set()
        elif event.input.id == "extra_args":
            self.run_worker(self.action_train())
        elif event.input.id == "kernel_path":
            self.run_worker(self.select_kernel(), group="kernel_select", exclusive=True)

    async def action_train(self):
        args = self.build_train_args()
        if args:
            await self.start_training(args)

    async def action_stop(self):
        self.run_worker(self.stop_training(), group="stop", exclusive=True)

    async def action_check(self):
        self.log_git("원격 확인")
        self.wake.set()

    async def action_force_sync(self):
        if self.training:
            self.notify("학습 중에는 강제 동기화할 수 없습니다. 먼저 중지하세요.", severity="warning")
            return
        async def job():
            await self.check_remote()
            if not await self.sync():
                self.log_git("이미 최신 상태입니다")
        self.run_worker(job(), group="sync", exclusive=True)

    async def action_quit(self):
        await self.stop_training()
        await self.shutdown_kernel()
        self.exit()

    # ---------------------------------------------------------------- notebook
    async def select_kernel(self, auto=False):
        field = self.query_one("#kernel_path", Input)
        if self.cells_busy:
            self.notify("셀 실행을 중단한 뒤 커널을 변경하세요.", severity="warning")
            return
        try:
            python = kernel_python_path(notebook_python() if auto else field.value)
            if python != self.kernel_python:
                code, _, err = await run(python, "-c", "import ipykernel_launcher", timeout=10)
                if code != 0:
                    raise ValueError(f"선택한 Python에서 ipykernel을 사용할 수 없습니다. "
                                     f'"{python}" -m pip install ipykernel\n{err}')
                # 검증하는 동안 셀 실행이 시작되었으면 기존 커널을 유지한다.
                if self.cells_busy:
                    self.notify("셀 실행을 중단한 뒤 커널을 변경하세요.", severity="warning")
                    return
                await self.shutdown_kernel()
                self.kernel_python = python
                self.runs = {}
                self.render_cell()
                self.log_git(f"노트북 커널 변경: {python} (다음 셀 실행 시 시작)", "green")
            field.value = python
            field.tooltip = f"선택한 커널 Python: {python}"
        except (OSError, ValueError) as exc:
            self.log_git(f"커널 선택 실패: {exc}", "red")
            self.notify(str(exc), title="커널 선택 실패", severity="error")

    def refresh_notebook_list(self):
        select = self.query_one("#nb_select", Select)
        notebooks = find_notebooks()
        current = self.nb_path if self.nb_path in notebooks else None
        with self.prevent(Select.Changed):
            select.set_options([(name, name) for name in notebooks])
            if current:
                select.value = current
        if self.nb_path and not current:
            self.log_git(f"노트북이 사라졌습니다: {self.nb_path}", "yellow")
            self.nb_path, self.cells = "", []
            self.render_cell()
        elif not self.nb_path and notebooks:
            select.value = notebooks[0]  # Select.Changed → load_notebook

    def load_notebook(self, rel_path, track_changes=False):
        """노트북을 (다시) 읽는다. track_changes면 이전 내용과 비교해 바뀐 셀을 표시한다."""
        try:
            cells = load_cells(rel_path)
        except (OSError, ValueError) as exc:
            self.log_git(f"노트북을 읽지 못했습니다: {rel_path}: {exc}", "red")
            return
        current_key = self.cells[self.cell_pos]["key"] if self.cells else None
        if track_changes:
            old = {cell["key"]: cell["source"] for cell in self.cells}
            self.changed_keys = {cell["key"] for cell in cells
                                 if old.get(cell["key"]) != cell["source"]}
            removed = len(set(old) - {cell["key"] for cell in cells})
            changed = [str(i + 1) for i, cell in enumerate(cells) if cell["key"] in self.changed_keys]
            if changed or removed:
                detail = f"바뀐 셀 {', '.join(changed)}" if changed else ""
                detail += f" 삭제 {removed}개" if removed else ""
                self.log_git(f"노트북 {rel_path} 갱신: {detail.strip()}", "cyan")
        else:
            self.changed_keys = set()
            self.runs = {}
            current_key = None
        self.nb_path, self.cells = rel_path, cells
        keys = [cell["key"] for cell in cells]
        self.cell_pos = (keys.index(current_key) if current_key in keys
                         else min(self.cell_pos, max(len(cells) - 1, 0)) if track_changes else 0)
        self.render_cell()

    def render_cell(self):
        panel = self.query_one("#notebook")
        kernel = ("없음" if self.kc is None else "실행 중" if self.cells_busy else "대기")
        panel.border_title = f"notebook · 커널 {kernel}"
        view = self.query_one("#nb_cell", Static)
        out = self.query_one("#cell_out_text", Static)
        self.update_output_title()
        if not self.cells:
            panel.border_subtitle = "0 / 0"
            view.update(Text("노트북이 없습니다." if not self.nb_path else "빈 노트북입니다.",
                             style="dim"))
            out.update(Text("셀 출력 없음", style="dim"))
            return
        cell = self.cells[self.cell_pos]
        run = self.runs.get(cell["key"])
        panel.border_subtitle = f"{self.cell_pos + 1} / {len(self.cells)}"
        header = Text()
        if cell["type"] == "code":
            header.append(f"In [{run.count if run and run.count else ' '}]", style="bold green")
        else:
            header.append(cell["type"], style="bold blue")
        if run:
            style = {"완료": "green", "에러": "red", "중단됨": "yellow"}.get(run.status, "cyan")
            header.append(f"  {run.status}", style=style)
        if cell["key"] in self.changed_keys:
            header.append("  ● 원격에서 변경됨", style="bold yellow")
            if run:
                header.append(" (출력은 이전 코드 기준)", style="dim")
        if cell["type"] == "markdown":
            body = Markdown(cell["source"])
        elif cell["type"] == "code":
            body = Syntax(cell["source"], "python", line_numbers=True, word_wrap=True,
                          background_color="default")
        else:
            body = Text(cell["source"])
        view.update(Group(header, body))
        if run and run.items:
            out.update(Group(header, Rule(style="dim"), *run.renderables()))
        elif run:
            out.update(Group(header, Text("(출력 없음)", style="dim")))
        else:
            out.update(Text("이 셀은 아직 실행하지 않았습니다.", style="dim"))

    # ------------------------------------------------------------------ output
    def show_output(self, mode):
        """출력 패널을 'train'(train.py) 또는 'cell'(현재 셀)로 전환한다."""
        self.query_one("#out_switch", ContentSwitcher).current = (
            "train_log" if mode == "train" else "cell_out")
        self.update_output_title()

    def update_output_title(self):
        output = self.query_one("#output")
        if self.query_one("#out_switch", ContentSwitcher).current == "train_log":
            state = "학습 중" if self.training else "대기"
            output.border_title = f"출력 · train.py ({state})"
        elif self.cells:
            output.border_title = f"출력 · 셀 {self.cell_pos + 1} / {len(self.cells)}"
        else:
            output.border_title = "출력 · 노트북"
        output.border_subtitle = "F4 전환"

    def action_toggle_output(self):
        current = self.query_one("#out_switch", ContentSwitcher).current
        self.show_output("cell" if current == "train_log" else "train")

    def on_select_changed(self, event: Select.Changed):
        if event.select.id != "nb_select" or event.value is Select.NULL:
            return
        if event.value == self.nb_path:
            return
        if self.cells_busy:
            self.notify("셀 실행 중에는 노트북을 바꿀 수 없습니다.", severity="warning")
            with self.prevent(Select.Changed):
                event.select.value = self.nb_path or Select.NULL
            return
        self.load_notebook(event.value)
        self.run_worker(self.shutdown_kernel(), group="kernel")  # 노트북마다 새 커널

    def action_move_cell(self, step):
        if self.cells:
            self.cell_pos = max(0, min(len(self.cells) - 1, self.cell_pos + step))
            self.query_one("#nb_view").scroll_home(animate=False)
            self.query_one("#cell_out").scroll_home(animate=False)
            self.show_output("cell")
            self.render_cell()

    @property
    def cells_busy(self):
        return self.exec_task is not None and not self.exec_task.done()

    async def action_run_cell(self):
        if self.cells:
            self.start_cells([self.cell_pos])
            self.action_move_cell(1)

    async def action_run_all(self):
        if self.cells:
            self.start_cells(list(range(self.cell_pos, len(self.cells))))

    def start_cells(self, indices):
        if AsyncKernelManager is None:
            self.notify("jupyter_client가 없습니다. build.py --setup-only로 환경을 구성하세요.",
                        severity="error")
            return
        if self.cells_busy:
            self.notify("이미 셀을 실행 중입니다.", severity="warning")
            return
        keys = [self.cells[i]["key"] for i in indices if self.cells[i]["type"] == "code"]
        if keys:
            self.show_output("cell")
            self.stop_queue = False
            self.exec_task = asyncio.create_task(self.run_cells(keys))

    async def run_cells(self, keys):
        try:
            await self.ensure_kernel()
            for key in keys:
                index = next((i for i, c in enumerate(self.cells) if c["key"] == key), None)
                if self.stop_queue or index is None:
                    break
                # 실행 중인 셀을 보고 있었다면 다음 셀로 화면을 따라간다.
                previous = next((i for i, c in enumerate(self.cells)
                                 if c["key"] == self.running_key), None)
                if previous is not None and previous == self.cell_pos:
                    self.cell_pos = index
                self.running_key = key
                if not await self.execute(key, self.cells[index]["source"]):
                    break
        except Exception as exc:  # noqa: BLE001 - 커널 오류를 화면에 보여준다
            self.log_git(f"노트북 실행 오류: {exc!r}", "red")
        finally:
            self.running_key = ""
            self.render_cell()

    async def ensure_kernel(self):
        async with self.kernel_lock:
            if self.kc is not None:
                return
            self.log_git(f"커널 시작 중… ({self.kernel_python})")
            km = AsyncKernelManager(kernel_name="python3")
            # 등록된 python3 kernelspec 대신 사용자가 선택한 인터프리터를 실행한다.
            km.kernel_spec.argv = [self.kernel_python, "-m", "ipykernel_launcher",
                                   "-f", "{connection_file}"]
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            # 커널의 stdout/stderr가 TUI 화면을 깨지 않도록 버린다(출력은 iopub로 받음).
            await km.start_kernel(cwd=str((ROOT / self.nb_path).parent), env=env,
                                  stdout=asyncio.subprocess.DEVNULL,
                                  stderr=asyncio.subprocess.DEVNULL)
            kc = km.client()
            kc.start_channels()
            try:
                await kc.wait_for_ready(timeout=60)
            except RuntimeError:
                kc.stop_channels()
                await km.shutdown_kernel(now=True)
                raise
            self.km, self.kc = km, kc
            self.log_git(f"커널 준비 완료 ({self.nb_path})", "green")

    async def execute(self, key, source):
        """셀 하나를 실행하고 출력을 모은다. 에러·중단이면 False."""
        run = self.runs[key] = CellRun()
        self.render_cell()
        msg_id = self.kc.execute(source, allow_stdin=False, stop_on_error=True)
        image_no = 0
        while True:
            try:
                msg = await self.kc.get_iopub_msg(timeout=1)
            except queue.Empty:
                if not await self.km.is_alive():
                    run.status = "커널 종료"
                    break
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "status" and content.get("execution_state") == "idle":
                break
            if kind == "execute_input":
                run.count = content.get("execution_count")
            elif kind == "stream":
                run.add_stream(content.get("name"), content.get("text", ""))
            elif kind in ("execute_result", "display_data"):
                if kind == "execute_result":
                    run.count = content.get("execution_count")
                data = content.get("data", {})
                if "image/png" in data:
                    image_no += 1
                    index = next(i for i, c in enumerate(self.cells) if c["key"] == key)                         if any(c["key"] == key for c in self.cells) else 0
                    target = (OUTPUT_DIR / Path(self.nb_path).stem /
                              f"cell{index + 1:03d}_{image_no}.png")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(base64.b64decode(data["image/png"]))
                    run.items.append(["rich", Text(f"[이미지] {target.relative_to(ROOT)}",
                                                   style="magenta")])
                elif "text/markdown" in data:
                    run.items.append(["rich", Markdown(data["text/markdown"])])
                elif "text/plain" in data:
                    run.items.append(["rich", Text.from_ansi(data["text/plain"])])
            elif kind == "error":
                run.status = "중단됨" if content.get("ename") == "KeyboardInterrupt" else "에러"
                run.items.append(["rich", Text.from_ansi("\n".join(content.get("traceback", [])))])
            elif kind == "clear_output":
                run.items.clear()
            if self.cells and self.cells[self.cell_pos]["key"] == key:
                self.render_cell()
        if run.status == "실행 중":
            run.status = "완료"
        self.render_cell()
        return run.status == "완료"

    async def action_interrupt_cells(self):
        await self.interrupt_cells()

    async def interrupt_cells(self):
        """대기 중인 셀을 취소하고 실행 중인 셀에 KeyboardInterrupt를 보낸다."""
        if not self.cells_busy:
            return
        self.stop_queue = True
        task = self.exec_task
        if self.km is not None:
            await self.km.interrupt_kernel()
        try:
            await asyncio.wait_for(asyncio.shield(task), 10)
        except TimeoutError:  # 인터럽트에 반응하지 않으면 실행 대기를 포기한다
            task.cancel()
            self.log_git("셀이 인터럽트에 반응하지 않아 대기를 취소했습니다", "yellow")
        except Exception:  # noqa: BLE001
            pass

    async def preempt_cells(self):
        """원격 변경 반영 전에 실행 중인 셀을 알리고 잠시 뒤 중단한다."""
        if not self.cells_busy:
            return
        message = f"원격 변경 반영을 위해 {PREEMPT_DELAY}초 뒤 셀 실행을 중단합니다"
        self.log_git(message, "yellow")
        self.notify(message, severity="warning", timeout=PREEMPT_DELAY + 1)
        self.bell()
        await asyncio.sleep(PREEMPT_DELAY)
        await self.interrupt_cells()

    async def action_restart_kernel(self):
        await self.interrupt_cells()
        await self.shutdown_kernel()
        self.runs = {}
        self.render_cell()
        self.log_git("커널을 종료했습니다. 다음 실행 때 새로 시작합니다")

    async def shutdown_kernel(self):
        async with self.kernel_lock:
            kc, km, self.kc, self.km = self.kc, self.km, None, None
            if kc is not None:
                kc.stop_channels()
            if km is not None:
                try:
                    await km.shutdown_kernel(now=True)
                except Exception:  # noqa: BLE001
                    pass


if __name__ == "__main__":
    Watchdog().run()
