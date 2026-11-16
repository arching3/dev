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

단축키: F2 학습 시작 · F3 학습 중지 · F5 지금 확인 · F6 강제 동기화 · Ctrl+Q 종료
"""

from __future__ import annotations

import asyncio
import codecs
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import signal
import sys

try:
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Footer, Input, Label, RichLog, SelectionList, Static
    from textual.widgets.selection_list import Selection
except ImportError:  # pragma: no cover
    sys.exit("textual이 필요합니다. `python build.py --setup-only`로 환경을 구성한 뒤 "
             ".venv의 Python으로 실행하거나 `pip install textual`을 실행하세요.")


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = "train.py"
DEFAULT_INTERVAL = 30  # 초
IS_WINDOWS = sys.platform == "win32"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
LIST_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+)$")


def now():
    return datetime.now().strftime("%H:%M:%S")


def venv_python():
    """build.py가 만든 .venv의 Python을 우선 사용하고, 없으면 현재 인터프리터를 쓴다."""
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    return str(candidate) if candidate.is_file() else sys.executable


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
    #train_log { height: 2fr; border: round $primary; }
    #progress { height: 1; padding: 0 1; color: $accent; }
    #git_log { height: 1fr; border: round $secondary; }
    """
    BINDINGS = [
        ("f2", "train", "학습 시작"),
        ("f3", "stop", "학습 중지"),
        ("f5", "check", "지금 확인"),
        ("f6", "force_sync", "강제 동기화"),
        ("ctrl+q", "quit", "종료"),
    ]

    def __init__(self):
        super().__init__()
        self.python = venv_python()
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
                yield RichLog(id="train_log", wrap=True, max_lines=5000, markup=False)
                yield Static(id="progress")
                yield RichLog(id="git_log", wrap=True, max_lines=1000, markup=False)
        yield Footer()

    async def on_mount(self):
        self.query_one("#train_log", RichLog).border_title = "train.py 출력"
        self.query_one("#git_log", RichLog).border_title = "git / watchdog"
        self.query_one("#experiments", SelectionList).border_title = "experiments"
        self.set_interval(1, self.render_status)
        self.log_git(f"Python: {self.python}")
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
        self.exit()


if __name__ == "__main__":
    Watchdog().run()
