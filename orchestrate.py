# -*- coding: utf-8 -*-
# Load .env before anything else
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)
except ImportError:
    pass

"""
Orchestration System v5 — Quantum Trading Council (TradingAgents-Enhanced)

에이전트 구성:
  Demis     (CEO, Opus)         — 회의 합성 → 팀장 태스크 배분 → 최종 보고서
  Radi      (Alpha Lead, Opus)  — 팀원 태스크 킥오프 → 결과 검토 → Demis 취합
  Casandra  (Beta Lead, Opus)   — 동상
  Viktor    (CTO, Opus)         — 수학적 알파 검증, OOS 방법론, 체제 이론 & 통계적 엄밀성
  팀원 8명  (codex exec, GPT)   — 실제 코딩 구현

흐름:
  0. 사용자: agenda + 문제 입력
  1. 회의: 사용자 ↔ Radi + Casandra + Viktor (멀티라운드)
  2. Demis: 회의 합성 → 팀장별 태스크 배분
  3. 팀 스프린트:
     a. 팀장 → 팀원에게 태스크 할당 (claude)
     b. 팀원 → codex exec 구현
     c. 팀장 → 결과 검토 (최대 2회 수정 요청)
     d. 팀장 → 전체 결과 취합 → Demis 전달
  4. Demis → 최종 보고서 사용자에게 제공

파일 네이밍: [TASK_NAME]-[WRITER]-[DEPARTMENT]-[DATE].md
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_FOLDER = Path("./project_output")
PROJECT_FOLDER.mkdir(exist_ok=True)
WORKSPACE = Path(__file__).parent
AGENDA_FILE = WORKSPACE / "agenda.md"


def _archive_agenda() -> None:
    """회의 종료 후 agenda.md를 날짜 버전으로 보관하고 원본을 비운다."""
    if not AGENDA_FILE.exists() or not AGENDA_FILE.read_text(encoding="utf-8").strip():
        return
    date_str  = datetime.now().strftime("%Y-%m-%d")
    archive   = WORKSPACE / f"agenda[Fin_Version_{date_str}].md"
    archive.write_text(AGENDA_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    AGENDA_FILE.write_text(
        "# Grand Council Agenda\n\n<!-- 새로운 안건을 여기에 추가하세요 -->\n",
        encoding="utf-8"
    )
    print(f"\n  [Agenda archived → {archive.name}]")

import platform as _platform
import google.generativeai as genai

# ── Research Tools (arXiv + OpenAlex + Crossref) ─────────────
try:
    from research_tools import search_papers, format_papers_for_prompt
    _RESEARCH_AVAILABLE = True
except ImportError:
    _RESEARCH_AVAILABLE = False
    def search_papers(*a, **k): return []
    def format_papers_for_prompt(*a, **k): return ""

# ── Cross-Session Memory (SQLite + ChromaDB) ─────────────────
try:
    from memory_layer import (
        save_session, save_agent_memory, update_task_status,
        get_agent_history, search_memory,
        format_agent_history, github_push_async, github_sync_to_db,
    )
    _MEMORY_LAYER_AVAILABLE = True
except ImportError:
    _MEMORY_LAYER_AVAILABLE = False
    def save_session(*a, **k): pass
    def save_agent_memory(*a, **k): pass
    def update_task_status(*a, **k): pass
    def get_agent_history(*a, **k): return []
    def search_memory(*a, **k): return []
    def format_agent_history(*a, **k): return ""
    def github_push_async(*a, **k): pass
    def github_sync_to_db(*a, **k): return 0

# ─────────────────────────────────────────────────────────────
# Gemini API 설정
# ─────────────────────────────────────────────────────────────
try:
    # IMPORTANT: Set the GEMINI_API_KEY environment variable.
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("\n  [Gemini API key loaded successfully]")
except (KeyError, TypeError):
    print("\n  [CRITICAL ERROR] GEMINI_API_KEY environment variable not found.")
    print("  Please set the 'GEMINI_API_KEY' environment variable for the script to function.")
    sys.exit(1)

# ── GitHub → SQLite sync at startup (background, non-blocking) ──
def _startup_github_sync():
    try:
        n = github_sync_to_db()
        if n > 0:
            print(f"  [Memory sync: {n} new session(s) imported from GitHub]")
    except Exception:
        pass
threading.Thread(target=_startup_github_sync, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# Checkpoint — persists session state across limit hits / restarts
# ─────────────────────────────────────────────────────────────
CHECKPOINT_FILE = PROJECT_FOLDER / ".checkpoint.json"

class Checkpoint:
    PHASES = ["adversarial_debate", "council", "synthesis", "plan_review",
              "alpha_sprint", "beta_sprint", "cto_sprint",
              "risk_debate", "final_report"]

    def __init__(self, data: dict = None):
        self.data = data or {
            "session_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "agenda": "", "problem": "",
            "phases_done": [],
            "transcript": "",
            "plan": "",
            "task_map": {},
            "approved_map": {},
            "member_results": {},   # key: "Lead:Member" → result text
            "summaries": {"alpha": "", "beta": "", "cto": ""},
            "adversarial_disputes": "",
            "risk_verdict": "",
        }

    def save(self):
        try:
            CHECKPOINT_FILE.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[CHECKPOINT] save failed: {e}", file=sys.stderr)

    def mark(self, phase: str, **kv):
        if phase not in self.data["phases_done"]:
            self.data["phases_done"].append(phase)
        for k, v in kv.items():
            self.data[k] = v
        self.save()

    def done(self, phase: str) -> bool:
        return phase in self.data.get("phases_done", [])

    def set_member(self, lead: str, member: str, result: str):
        self.data["member_results"][f"{lead}:{member}"] = result
        self.save()

    def get_member(self, lead: str, member: str) -> str | None:
        return self.data["member_results"].get(f"{lead}:{member}")

    @property
    def is_complete(self) -> bool:
        return "final_report" in self.data.get("phases_done", [])

    def clear(self):
        try:
            CHECKPOINT_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    @classmethod
    def load(cls) -> "Checkpoint | None":
        if not CHECKPOINT_FILE.exists():
            return None
        try:
            return cls(json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8")))
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────
# Session Memory — TradingAgents 패턴: 세션 간 결정 학습
# ─────────────────────────────────────────────────────────────
MEMORY_FOLDER = PROJECT_FOLDER / "memory"
MEMORY_FOLDER.mkdir(exist_ok=True)

class SessionMemory:
    """
    세션 간 영속 메모리.
    저장: (session_id, date, agenda_summary, key_decisions, sprint_outcomes)
    검색: 현재 agenda와 키워드 overlap 기반 유사 세션 top-k 반환
    → Grand Council 및 Demis synthesis에 주입
    """
    INDEX_FILE = MEMORY_FOLDER / "memory_index.json"

    def store(self, session_id: str, agenda: str, decisions: list,
              sprint_outcomes: dict, notes: str = "") -> None:
        entry = {
            "session_id": session_id,
            "date": datetime.now().isoformat()[:10],
            "agenda_summary": agenda[:300],
            "keywords": self._extract_keywords(agenda),
            "key_decisions": [str(d) for d in decisions[:10]],
            "sprint_outcomes": {k: str(v)[:200] for k, v in sprint_outcomes.items()},
            "notes": notes[:500],
        }
        index = self._load_index()
        index[session_id] = entry
        try:
            self.INDEX_FILE.write_text(
                json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[MEMORY] save failed: {e}", file=sys.stderr)

    def retrieve_similar(self, current_agenda: str, top_k: int = 3) -> str:
        """현재 agenda와 유사한 과거 세션 top-k → 프롬프트 주입용 문자열 반환"""
        index = self._load_index()
        if not index:
            return ""
        query_kw = set(self._extract_keywords(current_agenda))
        scored = []
        for sid, entry in index.items():
            overlap = len(query_kw & set(entry.get("keywords", [])))
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return ""
        lines = ["=== RELEVANT PAST SESSIONS (for context — learn from these) ==="]
        for score, entry in scored[:top_k]:
            decisions_str = "; ".join(entry.get("key_decisions", [])[:3])
            outcomes_str  = " | ".join(list(entry.get("sprint_outcomes", {}).values())[:2])
            lines.append(
                f"\n[{entry['date']}] {entry['agenda_summary'][:150]}\n"
                f"  Decisions: {decisions_str}\n"
                f"  Outcomes:  {outcomes_str[:200]}"
            )
        return "\n".join(lines)

    def _extract_keywords(self, text: str) -> list:
        important = {
            "QLSTM","FR","OI","CVD","EMA200","Gate","WR","BEP","OOS","ATR",
            "Structural","Feature","BTCUSDT","Koopman","Backtest","RL","MDD",
            "Sharpe","Alpha","Beta","Signal","Model","Retraining","Sprint",
            "SOLUSDT","ETHUSDT","Volatility","Regime","Liq","Cascade","Proxy",
        }
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_]{2,}\b', text)
        return list({w for w in words if w in important or w[0].isupper()})[:30]

    def _load_index(self) -> dict:
        if not self.INDEX_FILE.exists():
            return {}
        try:
            return json.loads(self.INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

_CP:  Checkpoint | None = None   # global checkpoint instance
_MEM: SessionMemory = SessionMemory()  # global memory instance

def _sigint_handler(_sig, _frame):
    print("\n\n  [Interrupted — saving checkpoint...]")
    if _CP:
        _CP.save()
        print(f"  [Checkpoint saved → {CHECKPOINT_FILE}]")
        # Sync GitHub push (not daemon — must complete before exit)
        try:
            import os as _os
            from github import Github as _Github
            _token = _os.environ.get("GITHUB_MEMORY_TOKEN")
            _repo_name = _os.environ.get("GITHUB_MEMORY_REPO")
            if _token and _repo_name:
                from datetime import datetime as _dt
                _sid = _CP.data.get("session_id", "interrupted")
                _content = f"[INTERRUPTED — partial session]\n\n{json.dumps(_CP.data, ensure_ascii=False, indent=2)}"
                _g = _Github(_token)
                _repo = _g.get_repo(_repo_name)
                _date = _dt.now().strftime("%Y-%m-%d_%H%M")
                _path = f"memory/sessions/{_date}_{_sid[:8]}_interrupted.md"
                _repo.create_file(_path, f"interrupted: {_sid[:8]}", _content.encode("utf-8"))
                print(f"  [GitHub push → {_path}]")
        except Exception as _e:
            print(f"  [GitHub push failed: {_e}]")
        print("  [Resume with: python orchestrate.py → r]")
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint_handler)


def _scan_last_session() -> dict:
    """
    Scan project_output for the most recent incomplete session.
    Returns a state dict with all artifacts found, empty dict if none.
    Incomplete = council Result.md exists but NO final_report for that date.
    """
    results = sorted(PROJECT_FOLDER.glob("*_Result.md"), reverse=True)
    for result_path in results:
        date_tag = result_path.stem[:10]  # "2026-03-21"
        finals = list(PROJECT_FOLDER.glob(f"final_report-Demis-Executive-{date_tag}.md"))
        if finals:
            continue  # session complete — skip

        state: dict = {
            "date_tag": date_tag,
            "result_path": result_path,
            "plan_path": None,
            "member_results": {},   # "Lead:Member" → text
            "summaries": {"alpha": "", "beta": "", "cto": ""},
            "phases_done": ["council"],
        }

        # Demis strategy plan
        plans = list(PROJECT_FOLDER.glob(f"demis_strategy_plan-Demis-Executive-{date_tag}.md"))
        if plans:
            state["plan_path"] = plans[0]
            state["phases_done"].append("synthesis")

        # Sprint completion reports
        for f in PROJECT_FOLDER.glob(f"sprint_completion_*-{date_tag}.md"):
            text = f.read_text(encoding="utf-8", errors="replace")
            if "Radi" in f.name or "Alpha" in f.name:
                state["summaries"]["alpha"] = text
                if "alpha_sprint" not in state["phases_done"]:
                    state["phases_done"].append("alpha_sprint")
            elif "Casandra" in f.name or "Beta" in f.name:
                state["summaries"]["beta"] = text
                if "beta_sprint" not in state["phases_done"]:
                    state["phases_done"].append("beta_sprint")

        # CTO validation report
        for f in PROJECT_FOLDER.glob(f"cto_validation_report-Viktor-*-{date_tag}.md"):
            state["summaries"]["cto"] = f.read_text(encoding="utf-8", errors="replace")
            if "cto_sprint" not in state["phases_done"]:
                state["phases_done"].append("cto_sprint")

        # Individual member implementation files (latest revision wins)
        # filename pattern: {name_lower}_{label}-{Name}-{Dept}-{date}.md
        all_members_map = {
            m["name"]: ("Radi" if m in ALPHA_MEMBERS else "Casandra")
            for m in (ALPHA_MEMBERS + BETA_MEMBERS)
        }
        for mname, lead in all_members_map.items():
            files = sorted(
                PROJECT_FOLDER.glob(f"{mname.lower()}_*-{mname}-*-{date_tag}.md")
            )
            if files:
                state["member_results"][f"{lead}:{mname}"] = \
                    files[-1].read_text(encoding="utf-8", errors="replace")

        return state
    return {}
_IS_WINDOWS = _platform.system() == "Windows"

# ─────────────────────────────────────────────────────────────
# 모델 & 에이전트
# ─────────────────────────────────────────────────────────────
MODEL_OPUS  = "gemini-2.5-pro"
MODEL_HAIKU = "gemini-2.5-pro" 

# ── 역할별 모델 티어 — 전체 gemini-2.5-pro ──
MODEL_TIER = {
    "council":         MODEL_OPUS,
    "adversarial":     MODEL_OPUS,
    "member_sprint":   MODEL_OPUS,
    "viktor_solo":     MODEL_OPUS,
    "demis_synthesis": MODEL_OPUS,
    "demis_report":    MODEL_OPUS,
    "risk_debate":     MODEL_OPUS,
    "lead_assign":     MODEL_OPUS,
    "lead_review":     MODEL_OPUS,
    "sprint_summary":  MODEL_OPUS,
    "signal_extract":  MODEL_OPUS,
}

DEMIS    = {"name": "Demis",    "codex": "CEO", "role": "Chief Executive Officer — 전략 합성 & 최종 보고", "model": MODEL_OPUS, "dept": "Executive"}
RADI     = {"name": "Radi",     "codex": "C1",  "role": "Team Alpha Lead — quant researcher & alpha signal design", "model": MODEL_OPUS, "dept": "Team Alpha"}
CASANDRA = {"name": "Casandra", "codex": "C2",  "role": "Team Beta Lead — systems architect & project lead",       "model": MODEL_OPUS, "dept": "Team Beta"}
VIKTOR   = {"name": "Viktor",   "codex": "C0",  "role": "Quant Researcher CTO — mathematical alpha validation, OOS methodology, regime theory & statistical rigor", "model": MODEL_OPUS, "dept": "CTO"}
LEADS    = [RADI, CASANDRA, VIKTOR]

ALPHA_MEMBERS = [
    {"name": "Darvin",  "codex": "C3",  "model": MODEL_OPUS, "role": "ML engineer — QLSTM architecture & training pipeline",         "dept": "Team Alpha"},
    {"name": "Felix",   "codex": "C4",  "model": MODEL_OPUS, "role": "data engineer — Bybit API, OHLCV pipeline, 13-dim features",   "dept": "Team Alpha"},
    {"name": "Jose",    "codex": "C5",  "model": MODEL_OPUS, "role": "risk manager — kill switch, position sizing, leverage control", "dept": "Team Alpha"},
    {"name": "Felipe",  "codex": "C6",  "model": MODEL_OPUS, "role": "backtest engineer — WR/MDD/Sharpe, OOS validation gates",      "dept": "Team Alpha"},
]
BETA_MEMBERS = [
    {"name": "Marvin",   "codex": "C7",  "model": MODEL_OPUS, "role": "quantum computing specialist — VQC circuits, PennyLane",       "dept": "Team Beta"},
    {"name": "Schwertz", "codex": "C8",  "model": MODEL_OPUS, "role": "trading strategy analyst — signal logic, regime detection",    "dept": "Team Beta"},
    {"name": "Finman",   "codex": "C9",  "model": MODEL_OPUS, "role": "performance engineer — CUDA optimization, GPU training speed", "dept": "Team Beta"},
    {"name": "Ilya",     "codex": "C10", "model": MODEL_OPUS, "role": "integration synthesizer — cross-team coordination",            "dept": "Team Beta"},
]
ALL_AGENTS = [DEMIS, RADI, CASANDRA, VIKTOR] + ALPHA_MEMBERS + BETA_MEMBERS


# ─────────────────────────────────────────────────────────────
# 파일 저장 유틸
# ─────────────────────────────────────────────────────────────
def _task_filename(task: str, writer: str, dept: str) -> str:
    """[TASK_NAME]-[WRITER]-[DEPARTMENT]-[DATE].md"""
    date     = datetime.now().strftime("%Y-%m-%d")
    task_slug = re.sub(r"[^\w]+", "_", task[:35]).strip("_").lower()
    dept_slug = dept.replace(" ", "")
    return f"{task_slug}-{writer}-{dept_slug}-{date}.md"


def _save(task: str, writer: str, dept: str, content: str) -> Path:
    fname = _task_filename(task, writer, dept)
    path  = PROJECT_FOLDER / fname
    path.write_text(content, encoding="utf-8")
    _dl(f"saved → {fname}")
    return path


# ─────────────────────────────────────────────────────────────
# 대시보드
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    name: str; codex: str; role: str; dept: str
    status: str = "IDLE"
    task: str = ""; preview: str = ""
    start_time: float = 0.0; elapsed: float = 0.0


class Dashboard:
    S_COLOR = {"IDLE": "dim", "ACTIVE": "bold green", "REVIEW": "bold yellow",
               "DONE": "blue", "ERROR": "red"}
    S_ICON  = {"IDLE": "○", "ACTIVE": "●", "REVIEW": "◎", "DONE": "✓", "ERROR": "✗"}

    def __init__(self):
        self.states: dict[str, AgentState] = {}
        self.lock = threading.Lock()
        self.phase = "STARTING"
        self.activity_log: list[str] = []
        self.council_chat: list[tuple[str, str]] = []
        self.council_mode = False
        self._live = None
        self._rich_available = False
        try:
            import rich; self._rich_available = True  # noqa
        except ImportError:
            pass

    def register(self, agents):
        for a in agents:
            self.states[a["name"]] = AgentState(
                name=a["name"], codex=a["codex"],
                role=a["role"], dept=a.get("dept","")
            )

    def update(self, name, *, status=None, task=None, msg=None, phase=None):
        with self.lock:
            if phase: self.phase = phase
            if name and name in self.states:
                s = self.states[name]
                if status:
                    prev = s.status
                    s.status = status
                    if status == "ACTIVE" and prev != "ACTIVE":
                        s.start_time = time.time(); s.elapsed = 0.0
                    elif status in ("DONE","ERROR","REVIEW") and s.start_time:
                        s.elapsed = time.time() - s.start_time
                if task:   s.task = task
                if msg:
                    lines = [l.strip() for l in msg.splitlines() if l.strip()]
                    if lines: s.preview = lines[-1][:60]
                    ts = datetime.now().strftime("%H:%M:%S")
                    c  = self.S_COLOR.get(s.status,"white")
                    ic = self.S_ICON.get(s.status,"?")
                    self.activity_log.append(
                        f"[dim]{ts}[/] [{c}]{ic} {name}[/]: {lines[-1][:65] if lines else ''}")
                    self.activity_log = self.activity_log[-20:]

    def log_event(self, msg):
        with self.lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.activity_log.append(f"[dim]{ts}[/] [yellow]{msg}[/]")
            self.activity_log = self.activity_log[-20:]

    def add_council_msg(self, speaker, msg):
        with self.lock:
            self.council_chat.append((speaker, msg))
            self.council_chat = self.council_chat[-60:]

    # ── 렌더링 ─────────────────────────────────────────────
    def _card(self, s: AgentState, h=7):
        from rich.panel import Panel
        c  = self.S_COLOR.get(s.status, "white")
        ic = self.S_ICON.get(s.status, "?")
        el = ""
        if s.status == "ACTIVE" and s.start_time: el = f" [dim]{time.time()-s.start_time:.0f}s[/]"
        elif s.status in ("DONE","REVIEW") and s.elapsed: el = f" [dim]{s.elapsed:.0f}s[/]"
        t  = (s.task[:33]    + "…") if len(s.task)    > 33 else s.task
        p  = (s.preview[:43] + "…") if len(s.preview) > 43 else s.preview
        body = (f"[{c}]{ic} {s.status}[/]{el}\n"
                f"[dim italic]{s.role[:38]}[/]\n{t}\n[italic dim]{p}[/]")
        bd = "green" if s.status == "ACTIVE" else ("yellow" if s.status == "REVIEW" else
             "blue" if s.status == "DONE" else "grey50")
        return Panel(body, title=f"[bold]{s.name}[/] [dim]{s.codex}[/]",
                     title_align="left", border_style=bd, height=h)

    def _render(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.console import Group

        active_n = sum(1 for s in self.states.values() if s.status == "ACTIVE")
        done_n   = sum(1 for s in self.states.values() if s.status == "DONE")
        mode_tag = "[bold magenta]◆ GRAND COUNCIL[/]" if self.council_mode else ""
        header = Panel(
            f"[bold cyan]QUANTUM TRADING ORCHESTRATION[/]  ·  "
            f"[yellow]{self.phase}[/]  {mode_tag}  ·  "
            f"[green]active {active_n}[/]  [blue]done {done_n}[/]  total {len(self.states)}",
            style="on grey11", height=3)

        # 에이전트 그리드: CEO(1행), Leads(2행), Members(3-4행) — 5열
        ags = list(self.states.values())
        grid = Table.grid(expand=True)
        for _ in range(5): grid.add_column(ratio=1)
        empty = lambda: Panel("", height=7, border_style="grey23")

        # row 0: CEO only
        grid.add_row(self._card(ags[0]), empty(), empty(), empty(), empty())
        # row 1: leads (Radi, Casandra, Viktor)
        n_leads = len(LEADS)
        lead_cells = [self._card(ags[1 + i]) for i in range(n_leads)]
        while len(lead_cells) < 5: lead_cells.append(empty())
        grid.add_row(*lead_cells)
        # rows 2-3: 8 members (4+4)
        member_start = 1 + n_leads
        for chunk in [ags[member_start:member_start+4], ags[member_start+4:member_start+8]]:
            cells = [self._card(s) for s in chunk]
            while len(cells) < 5: cells.append(empty())
            grid.add_row(*cells)

        # 하단 패널
        if self.council_mode:
            SPKR = {"YOU":"bold yellow","Radi":"bold green","Casandra":"bold cyan","Viktor":"bold blue","Demis":"bold magenta","Jose":"bold red","Council":"bold white"}
            lines = []
            for spk, msg in self.council_chat[-14:]:
                clr   = SPKR.get(spk, "white")
                short = msg[:300].replace("\n"," ")
                if len(msg) > 300: short += "…"
                lines.append(f"[{clr}][{spk}][/]  {short}")
            bottom = Panel("\n\n".join(lines) or "[dim]대화 시작 전…[/]",
                           title="[bold magenta]◆ Grand Council Chat[/]",
                           border_style="magenta", height=22)
        else:
            log_text = "\n".join(self.activity_log[-8:]) or "[dim]Waiting…[/]"
            bottom = Panel(log_text, title="[bold]Activity Log[/]",
                           border_style="grey50", height=12)

        return Group(header, grid, bottom)

    # ── renderable proxy (Rich이 필요할 때만 _render 호출) ──
    def _make_renderable(self):
        dash = self
        class _Proxy:
            def __rich_console__(self, console, options):
                yield dash._render()
        return _Proxy()

    def pause(self):
        if self._live:
            try: self._live.stop()
            except Exception: pass

    def resume(self):
        if not self._rich_available: return
        from rich.live import Live
        self._live = Live(
            self._make_renderable(),
            refresh_per_second=0.2,
            screen=False,
            transient=False,
            auto_refresh=True,
        )
        self._live.__enter__()

    def __enter__(self):
        if self._rich_available:
            from rich.live import Live
            self._live = Live(
                self._make_renderable(),
                refresh_per_second=0.2,
                screen=False,
                transient=False,
                auto_refresh=True,
            )
            self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            try: self._live.__exit__(*args)
            except Exception: pass


# ── 전역 대시보드 ───────────────────────────────────────────
_DASH: Optional[Dashboard] = None
def _du(name, **kw):
    if _DASH: _DASH.update(name, **kw)
def _dl(msg):
    if _DASH: _DASH.log_event(msg)
    else: print(msg)
def _dc(speaker, msg):
    if _DASH: _DASH.add_council_msg(speaker, msg)


# ─────────────────────────────────────────────────────────────
# 에이전트 호출 — Gemini API
# ─────────────────────────────────────────────────────────────
# Safety settings: allow technical/financial code generation without blocks
_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT",          "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",         "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",   "threshold": "BLOCK_NONE"},
]

# Max prompt chars before truncation (~150K tokens for gemini-2.5-pro 1M ctx)
_MAX_PROMPT_CHARS = 600_000


def call_gemini(agent: dict, prompt: str, model_name: str, timeout: int,
                allow_search: bool = False) -> str:
    """Calls the Gemini API with the given prompt and model.
    If allow_search=True, enables Google Search grounding so the model can
    look up papers, documentation, and internet resources in real time.
    """
    name = agent["name"]
    _du(name, status="ACTIVE", msg="Generating with Gemini...")

    # Context overflow guard: truncate middle of prompt if too large
    if len(prompt) > _MAX_PROMPT_CHARS:
        keep_head = _MAX_PROMPT_CHARS // 3
        keep_tail = _MAX_PROMPT_CHARS - keep_head - 200
        prompt = (prompt[:keep_head]
                  + f"\n\n[...TRUNCATED {len(prompt) - keep_head - keep_tail} chars...]\n\n"
                  + prompt[-keep_tail:])
        _dl(f"[CONTEXT] {name}: prompt truncated to {len(prompt)} chars (500 overflow guard)")

    try:
        # glm.Tool(google_search=...) — gemini-2.5-pro 지원 포맷
        if allow_search:
            import google.ai.generativelanguage as _glm
            tools = [_glm.Tool(google_search=_glm.Tool.GoogleSearch())]
        else:
            tools = None
        model = genai.GenerativeModel(model_name, tools=tools,
                                      safety_settings=_SAFETY_SETTINGS)
        # Gemini's 'timeout' in generate_content is a float in seconds
        response = model.generate_content(
            prompt,
            request_options={'timeout': float(timeout)}
        )

        # Handle cases where the response might be empty or blocked
        if not response.parts:
            result_text = "[GEMINI] Response was blocked or empty."
            _du(name, status="ERROR", msg=result_text)
            return result_text

        result_text = response.text.strip()
        if not result_text:
            result_text = "[GEMINI] No text response."

        _du(name, status="DONE", msg=result_text[:120])
        return result_text

    except Exception as e:
        err_str = str(e)
        # Rate limit / quota → fallback to gemini-2.5-flash
        if any(k in err_str for k in ("429", "quota", "RESOURCE_EXHAUSTED", "rate")):
            fallback = "gemini-2.5-flash"
            if model_name != fallback:
                _du(name, status="ACTIVE", msg=f"Rate limit — retrying with {fallback}...")
                return call_gemini(agent, prompt, fallback, timeout, allow_search)
        # 500 Internal Server Error — usually context still too large after truncation
        if "500" in err_str or "Internal" in err_str:
            if len(prompt) > 100_000:
                truncated = prompt[:50_000] + "\n\n[HARD TRUNCATED]\n\n" + prompt[-50_000:]
                _dl(f"[500] {name}: hard truncation to 100K chars and retry")
                return call_gemini(agent, truncated, model_name, timeout, allow_search)
        error_msg = f"[GEMINI ERROR] {type(e).__name__}: {str(e)}"
        print(f"\nError details for {name}: {error_msg}", file=sys.stderr)
        _du(name, status="ERROR", msg=error_msg[:120])
        return error_msg


# ─────────────────────────────────────────────────────────────
# ReAct Loop — parse Action: tags, execute, feed Observation:
# ─────────────────────────────────────────────────────────────
_REACT_ACTION_RE = re.compile(
    r"Action:\s*(\w+)\s*\(\s*(.*?)\s*\)\s*$", re.MULTILINE | re.DOTALL
)
_MAX_REACT_ITERS = 6


def _exec_react_action(action: str, args_raw: str) -> str:
    """Execute a ReAct tool call and return the observation string."""
    # Strip surrounding quotes from first arg
    def _unquote(s: str) -> str:
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    action = action.strip()
    try:
        if action in ("Read", "read_file"):
            path = _unquote(args_raw)
            p = Path(path) if Path(path).is_absolute() else WORKSPACE / path
            if p.exists():
                content = p.read_text(encoding="utf-8", errors="replace")
                return content[:8000] + ("\n[...truncated...]" if len(content) > 8000 else "")
            return f"[File not found: {p}]"

        elif action in ("Write", "write_file"):
            # args: 'path', 'content'
            parts = args_raw.split(",", 1)
            path = _unquote(parts[0])
            content = _unquote(parts[1].strip()) if len(parts) > 1 else ""
            p = Path(path) if Path(path).is_absolute() else WORKSPACE / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"[Written {len(content)} chars to {p}]"

        elif action in ("Bash", "bash", "run"):
            cmd = _unquote(args_raw)
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                cwd=str(WORKSPACE), timeout=60
            )
            out = (result.stdout + result.stderr).strip()
            return out[:4000] + ("\n[...truncated...]" if len(out) > 4000 else "") or "[no output]"

        elif action in ("Glob", "glob_files"):
            import glob as _glob
            pattern = _unquote(args_raw)
            base = str(WORKSPACE) + "/"
            matches = _glob.glob(base + pattern, recursive=True)
            return "\n".join(matches[:50]) or "[no matches]"

        elif action in ("Grep", "grep"):
            parts = args_raw.split(",", 1)
            pattern = _unquote(parts[0])
            path = _unquote(parts[1].strip()) if len(parts) > 1 else "."
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", pattern, path],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", cwd=str(WORKSPACE), timeout=30
            )
            out = result.stdout.strip()
            return out[:3000] or "[no matches]"

        else:
            return f"[Unknown action: {action}]"

    except subprocess.TimeoutExpired:
        return "[Action timed out]"
    except Exception as exc:
        return f"[Action error: {exc}]"


def call_gemini_react(agent: dict, prompt: str, model_name: str,
                      timeout: int) -> str:
    """ReAct loop: call Gemini, parse Action: tags, execute tools,
    feed Observation: back until Final Answer: or max iterations."""
    name = agent["name"]
    conversation = prompt  # grows with each Thought/Action/Observation turn

    for iteration in range(_MAX_REACT_ITERS):
        _dl(f"[ReAct] {name} iter {iteration+1}/{_MAX_REACT_ITERS}")
        response = call_gemini(agent, conversation, model_name, timeout,
                               allow_search=False)

        # If error or blocked → return as-is
        if response.startswith("[GEMINI"):
            return response

        conversation += f"\n\n{response}"

        # Check for Final Answer
        if "Final Answer" in response or "final answer" in response.lower():
            return response

        # Parse all Action: lines
        actions = _REACT_ACTION_RE.findall(response)
        if not actions:
            # No more actions — model is done
            return response

        # Execute each action and append observations
        obs_block = ""
        for act_name, act_args in actions:
            obs = _exec_react_action(act_name, act_args)
            obs_block += f"\nObservation: {obs}\n"
            _dl(f"[ReAct] {name} — {act_name}(...) → {obs[:80]}...")

        conversation += obs_block + "\nThought:"

    # Max iters reached — return last response
    return response


def call_claude(agent: dict, system_prompt: str, user_prompt: str,
                timeout: int = 120, allow_tools: bool = False,
                session_id: str = None, tier: str = None) -> str:
    """
    Prepares prompt and calls the Gemini API.
    tier: key into MODEL_TIER for role-based model selection (Pro vs Flash).
    allow_tools=True enables Google Search grounding.
    session_id is ignored (Gemini stateless).
    """
    model = MODEL_TIER.get(tier, agent["model"]) if tier else agent["model"]
    payload = (f"{system_prompt}\n\n---\n\n{user_prompt}"
               if system_prompt else user_prompt)

    return call_gemini(agent, payload, model, timeout, allow_search=allow_tools)


def call_codex(member: dict, prompt: str, timeout: int = 300) -> str:
    """
    (Replaced) Calls the Gemini API for code generation/implementation tasks.
    """
    agent_dict = {
        "name": member.get("name", "Codex"),
        "model": MODEL_OPUS,
    }
    return call_gemini(agent_dict, prompt, MODEL_OPUS, timeout)



# ─────────────────────────────────────────────────────────────
# Part B — Execution Task Detection + Claude Code CLI Executor
# ─────────────────────────────────────────────────────────────
_EXEC_KEYWORDS = {
    "run", "execute", "backtest", "validate", "test", "measure",
    "script", "result", "gate", "check", "verify", "calculate",
    "compute", "analyze", "analyse", "benchmark", "evaluate",
    "perform", "conduct", "output", "generate report",
}

def _is_execution_task(task: str) -> bool:
    """Return True if task requires actual script execution (not just code writing)."""
    t = task.lower()
    return any(kw in t for kw in _EXEC_KEYWORDS)


_CLI_LIMIT_SIGNALS = (
    "rate limit", "429", "overloaded", "quota", "limit reached",
    "too many requests", "usage limit", "capacity", "you've hit your limit",
)

def call_claude_cli(prompt: str, cwd=None, timeout: int = 300) -> str:
    """Claude Code CLI (claude --print) subprocess — real Bash/file tool execution.
    Falls back to Gemini 2.5 Pro if Claude hits rate/usage limits.
    """
    try:
        result = subprocess.run(
            ["claude", "--print", "--dangerously-skip-permissions", "--model", "claude-opus-4-6"],
            input=prompt,           # stdin으로 전달 (CLI 인자 한도 우회)
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd or WORKSPACE),
            timeout=timeout,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()

        # Check stdout + stderr for limit signals → fallback to Gemini
        combined_out = (out + " " + err).lower()
        if any(sig in combined_out for sig in _CLI_LIMIT_SIGNALS):
            _dl(f"[CLI→GEMINI] Claude limit detected — falling back to gemini-2.5-pro")
            return call_gemini(
                {"name": "CLI-fallback"}, prompt, MODEL_OPUS, timeout, allow_search=False
            )

        if out:
            return out
        if err:
            return f"[STDERR] {err}"
        return "[NO OUTPUT]"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s]"
    except FileNotFoundError:
        # claude CLI not installed → fallback to Gemini directly
        _dl("[CLI→GEMINI] claude CLI not found — falling back to gemini-2.5-pro")
        # Strip tool-call instructions so Gemini doesn't hallucinate glob_files/read_file
        _TOOL_PHRASES = (
            "glob_files(", "read_file(", "bash(", "MANDATORY FIRST STEP",
            "Start by exploring project_output/ with glob_files",
            "You have FULL READ ACCESS to all project files via tools",
        )
        clean_prompt = "\n".join(
            line for line in prompt.splitlines()
            if not any(p in line for p in _TOOL_PHRASES)
        )
        return call_gemini(
            {"name": "CLI-fallback"}, clean_prompt, MODEL_OPUS, timeout, allow_search=False
        )
    except Exception as e:
        return f"[ERROR: {e}]"


def call_claude_exec(member: dict, system_prompt: str, user_prompt: str,
                     timeout: int = 600) -> str:
    """Part B: Claude Code CLI subprocess — real Bash/Read/Write/Glob tool execution."""
    combined = f"[SYSTEM ROLE]\n{system_prompt}\n\n[TASK]\n{user_prompt}"
    _dl(f"[CLI] {member.get('name','?')} → claude CLI: {user_prompt[:80]}...")
    return call_claude_cli(combined, cwd=WORKSPACE, timeout=timeout)



# ─────────────────────────────────────────────────────────────
# Phase 0: 사용자 의제 입력
# ─────────────────────────────────────────────────────────────
def _is_empty_agenda(text: str) -> bool:
    """Returns True if text is just the empty agenda template with no real content."""
    s = text.strip()
    return not s or ("새로운 안건을 여기에 추가하세요" in s and len(s) < 120)


def _load_md(path_str: str) -> str | None:
    """파일 경로면 내용 반환. 빈 agenda 템플릿이면 None."""
    p = Path(path_str.strip().strip('"').strip("'"))
    if not p.is_absolute():
        p = WORKSPACE / p
    if p.exists() and p.suffix.lower() in (".md", ".txt", ""):
        text = p.read_text(encoding="utf-8")
        return None if _is_empty_agenda(text) else text
    return None


def get_user_agenda() -> tuple[str, str]:
    SEP = "═" * 70
    print(f"\n{SEP}")
    print("  QUANTUM TRADING ORCHESTRATION — 회의 설정")
    print(SEP)

    # ── 안건 입력 ───────────────────────────────────────────
    # _load_md already returns None for empty template
    _auto_md = _load_md(str(AGENDA_FILE)) if AGENDA_FILE.exists() else None
    if _auto_md:
        print(f"\n[AGENDA] agenda.md 자동 로드됨 ({len(_auto_md)}자)")
        print("  · Enter — 그대로 사용")
        print("  · 다른 파일 경로 또는 직접 입력 — 덮어쓰기")
        agenda_input = input("  Agenda [Enter = agenda.md]: ").strip()
        if not agenda_input:
            md_content = _auto_md
        else:
            md_content = _load_md(agenda_input) if agenda_input.endswith(".md") else None
            if not md_content:
                md_content = None
    else:
        print("\n[AGENDA] 이번 회의 안건을 입력하세요.")
        print("  · 직접 입력  예) B+C 병행 전략 구현 계획")
        print("  · .md 파일   예) agenda.md  또는  C:/path/to/file.md")
        agenda_input = input("  Agenda: ").strip()
        md_content = _load_md(agenda_input) if agenda_input else None

    if md_content:
        # 파일 첫 번째 # 제목을 agenda, 나머지를 problem으로 분리
        lines = md_content.splitlines()
        title_lines = [l for l in lines if l.startswith("# ")]
        agenda  = title_lines[0].lstrip("# ").strip() if title_lines else agenda_input
        problem = md_content   # 전체 파일 내용을 problem context로 사용
        print(f"\n  [파일 로드 완료] {agenda_input}")
        print(f"  Agenda  : {agenda}")
        print(f"  Content : {len(problem)}자 ({problem.count(chr(10))+1}줄)")
    else:
        agenda = agenda_input or "B+C 병행 전략 구현 계획"

        # ── 문제 입력 ─────────────────────────────────────
        print("\n[PROBLEM] 해결해야 할 문제를 입력하세요.")
        print("  · 직접 입력 후 빈 줄 Enter → 완료")
        print("  · .md 파일   예) agenda.md  또는  C:/path/to/file.md")
        first = input("  > ").strip()

        md2 = _load_md(first) if first else None
        if md2:
            problem = md2
            print(f"\n  [파일 로드 완료] {first} — {len(problem)}자")
        else:
            prob_lines = [first] if first else []
            while True:
                line = input("  > ")
                if line == "": break
                prob_lines.append(line)
            problem = "\n".join(prob_lines) if prob_lines else (
                "1. FR+EMA200 알파: BTC 전용, 월 3회 (WR=32.2%)\n"
                "2. QLSTM OOS WR=26.4% → 13-dim 구조적 피처로 재훈련 필요\n"
                "3. 모델 단독 패스(p_long>0.70)를 제2 알파로 활용 가능한가?"
            )

    print(f"\n{SEP}")
    print(f"  Agenda  : {agenda}")
    print("  Problem :")
    for l in problem.splitlines()[:8]:
        print(f"    {l}")
    if problem.count("\n") >= 8:
        print(f"    … (총 {problem.count(chr(10))+1}줄)")
    print(SEP)
    input("\n  [Enter]를 눌러 회의를 시작하세요...")
    return agenda, problem


# ─────────────────────────────────────────────────────────────
# Phase 1: Grand Council — Viktor-led structured debate
# ─────────────────────────────────────────────────────────────

# Council order: Viktor leads
COUNCIL_ORDER = [VIKTOR, RADI, CASANDRA]

# Attack pairs: each agent attacks each other agent once per round (6 pairs)
# Viktor goes first in all orderings
ATTACK_PAIRS = [
    (VIKTOR,   RADI),
    (VIKTOR,   CASANDRA),
    (RADI,     VIKTOR),
    (RADI,     CASANDRA),
    (CASANDRA, VIKTOR),
    (CASANDRA, RADI),
]

# Shared project context injected into every system prompt (compact)
_PROJ_CTX = (
    "Project: BTCUSDT 1h, eff_leverage=5x, BEP=25.4%, TP=3xATR/SL=1xATR, fees=0.375%/trade\n"
    "Baseline: FR+EMA200 WR=36.8% ROI=+126% MDD=16.27% (2023-2026)\n"
    "Failed: 28-dim statistical QLSTM OOS WR=26.4% ROI=-54% -- generalization failure confirmed\n"
    "Goal: 13-dim structural feature QLSTM + model-only alpha validation (Gate 1: WR>=30% on 2026 Q1)"
)

# Files whose code digest is injected into lead prompts
_COUNCIL_FILES = [
    "src/models/features_structural.py",
    "scripts/backtest_structural.py",
    "scripts/backtest_behavioral.py",
    "src/models/integrated_agent.py",
    "src/models/features_v3.py",
]

def _build_code_digest(max_chars_per_file: int = 1800) -> str:
    """
    Extract compact code digest from key files:
    module docstring + class/function signatures + top-level constants.
    Agents get the actual structure without reading every line body.
    """
    import ast, textwrap

    def _digest_file(path: Path) -> str:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

        lines = src.splitlines()
        kept = []

        # Module docstring (first triple-quoted string)
        if lines and lines[0].startswith('"""'):
            for i, l in enumerate(lines):
                kept.append(l)
                if i > 0 and '"""' in l:
                    break

        # Class defs, function signatures, top-level constants
        in_func_body = False
        indent_level = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # class definition
            if stripped.startswith("class "):
                kept.append(line)
                continue
            # function/method signature (keep def line + docstring line if present)
            if stripped.startswith("def ") or stripped.startswith("async def "):
                kept.append(line)
                # include first docstring line if next non-empty line is it
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if nxt.startswith('"""') or nxt.startswith("'''"):
                        kept.append(lines[i + 1])
                continue
            # top-level constants (ALL_CAPS = ... or typed hints)
            if not line.startswith(" ") and not line.startswith("\t"):
                if "=" in stripped and stripped[0].isupper():
                    kept.append(line)

        digest = "\n".join(kept)
        # truncate to budget
        if len(digest) > max_chars_per_file:
            digest = digest[:max_chars_per_file] + "\n# ... (truncated)"
        return digest

    sections = []
    for rel in _COUNCIL_FILES:
        full = WORKSPACE / rel
        if not full.exists():
            continue
        digest = _digest_file(full)
        if digest.strip():
            sections.append(f"=== {rel} ===\n{digest}")

    return "\n\n".join(sections)


# Built once at import time; injected into every council call
_CODE_DIGEST:    str = ""   # populated in run_grand_council()
_ADV_CONTEXT:    str = ""   # pre-council adversarial debate summary
_MEMORY_CONTEXT: str = ""   # similar past sessions from SessionMemory

_PERSONAS = {
    "Viktor": (
        "You are Viktor, Quant Researcher CTO. You chair this council and set its intellectual direction. "
        "You enforce rigorous OOS methodology, calibration standards, and statistical validity. "
        "You are aware of the latest academic research in quantitative finance, machine learning, and quantum computing — "
        "you cite specific papers (author, year, title) when they support or refute a claim. "
        "You challenge every unsubstantiated claim with formal arguments and propose the mathematically sound path. "
        "When you attack, you identify the precise flaw in an argument. When you defend, you cite derivations, not opinions."
    ),
    "Radi": (
        "You are Radi, Team Alpha Lead. You are the strongest advocate for structural market microstructure signals. "
        "You defend the 13-dim feature set (FR, OI, CVD, liquidation) with empirical WR/EV numbers. "
        "You are aware of the latest alpha research — Lopez de Prado, Easley, Cont, Hawkes process literature — "
        "and cite specific papers when proposing new alpha signals or defending existing ones. "
        "You are skeptical of overfitting and fight back hard when your alpha signals are challenged. "
        "When you attack, you expose gaps in implementation or statistical assumptions."
    ),
    "Casandra": (
        "You are Casandra, Team Beta Lead. You care about one thing: can we actually build and validate this? "
        "You are aware of engineering best practices and recent MLOps / RL systems papers. "
        "You challenge vague proposals with concrete failure modes, missing dependencies, and timeline reality. "
        "When you attack, you force the other person to be specific: file path, function name, test criterion. "
        "When you defend, you ground your position in engineering constraints, not theory."
    ),
}

_COUNCIL_RULES = (
    "STRICT RULES — no exceptions:\n"
    "1. English only. Plain text. No markdown, no headers, no bullet points, no bold, no asterisks.\n"
    "2. Never restate the agenda, project context, or numbers everyone already knows.\n"
    "3. Max 4 sentences. Be dense and direct.\n"
    "4. No greetings, no 'I think', no filler. Start with your actual point."
)



def _council_call(lead: dict, system: str, user: str) -> str:
    """
    Fresh-session council call.
    system + code digest + adversarial context + memory merged into system_prompt.
    """
    code_section = (
        f"\n\nACTUAL CODEBASE (live -- reference exact names/structures below):\n"
        f"{_CODE_DIGEST}"
    ) if _CODE_DIGEST else ""
    full_system = f"{system}{code_section}{_ADV_CONTEXT}{_MEMORY_CONTEXT}"
    return call_claude(lead, full_system, user, timeout=300, session_id=None,
                       tier="council")


def _opening_statement(lead: dict, agenda: str, problem: str) -> str:
    system = (
        f"{_PERSONAS[lead['name']]}\n\n"
        f"{_PROJ_CTX}\n\n"
        f"{_COUNCIL_RULES}"
    )
    # Strip markdown symbols from agenda to reduce mirroring
    agenda_clean = re.sub(r"[*#`|]+", "", agenda).strip()
    user = (
        "IMPORTANT: Your response must be in English only, plain text, no markdown. "
        "Do not summarize or repeat what follows. Read it once, then state your own expert position.\n\n"
        f"AGENDA:\n{agenda_clean[:1200]}\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"You are {lead['name']}. In 3-4 English sentences, state your single most critical "
        "concern or proposal. Be assertive and specific — cite a number or a technical claim."
    )
    return _council_call(lead, system, user)


def _challenge_statement(challenger: dict, target: dict,
                         target_last: str, round_n: int) -> str:
    """One expert challenges another's technical position."""
    system = (
        f"{_PERSONAS[challenger['name']]}\n\n"
        f"{_PROJ_CTX}\n\n"
        f"{_COUNCIL_RULES}"
    )
    # Strip markdown/special chars from target's statement to avoid CMD issues
    target_clean = re.sub(r"[*#`\"\|<>^&]+", "", target_last).strip()[:350]
    user = (
        f"Round {round_n}. You are {challenger['name']} critiquing {target['name']}.\n\n"
        f"{target['name']} stated: {target_clean}\n\n"
        f"Identify the single weakest or most flawed claim in that statement. "
        f"Deliver a sharp, specific technical counter-argument. "
        f"English only, plain text, 3-4 sentences. No questions — make an assertive critique."
    )
    return _council_call(challenger, system, user)


def _rebuttal_statement(defender: dict, challenger: dict,
                        challenge_msg: str, round_n: int) -> str:
    system = (
        f"{_PERSONAS[defender['name']]}\n\n"
        f"{_PROJ_CTX}\n\n"
        f"{_COUNCIL_RULES}"
    )
    challenge_clean = re.sub(r"[*#`\"\|<>^&]+", "", challenge_msg).strip()[:350]
    user = (
        f"Round {round_n}. {challenger['name']} critiqued your position:\n\n"
        f"{challenge_clean}\n\n"
        f"Rebut in 3-4 English sentences. Either defend with new evidence "
        f"or concede the point and pivot to a stronger claim. No evasion. Plain text only."
    )
    return _council_call(defender, system, user)


def _extract_decisions(history: list, existing: list) -> list:
    new_decisions = list(existing)
    keywords = ["agree", "confirmed", "we should", "the threshold", "decided",
                "Gate 1", "Gate 2", "implement", "reject", "fallback", "settle on",
                "conclude", "accept", "the fix is"]
    for spk, msg in history[-8:]:
        if spk in ("SYSTEM", "YOU"):
            continue
        for kw in keywords:
            if kw.lower() in msg.lower():
                first_sent = msg.strip().split(".")[0][:120]
                entry = f"[{spk}] {first_sent}"
                if entry not in new_decisions:
                    new_decisions.append(entry)
                break
    return new_decisions[-12:]


# ─────────────────────────────────────────────────────────────
# Phase 0.5: Pre-Council Results Scan (data-driven, replaces BULL/BEAR debate)
# ─────────────────────────────────────────────────────────────
def run_adversarial_debate(agenda: str, problem: str) -> dict:
    """
    Grand Council 전 데이터 기반 현황 스캔.
    토론 대신 실제 백테스트 결과와 Gate 상태를 읽어 컨텍스트 생성.
    Radi: 지난 백테스트 결과 요약 / Viktor: 수학적 게이트 상태 감사
    2 API calls (debate 6 → 2로 축소)
    """
    _DASH and _DASH.update("", phase="Pre-Council · Results Scan")
    _dl("Pre-Council Results Scan: Radi(backtest) + Viktor(gate audit) — data-driven")

    # Radi — 지난 백테스트 결과 읽기
    radi_sys = (
        f"{_PERSONAS['Radi']}\n\n{_PROJ_CTX}\n\n"
        "ROLE: You are scanning the latest backtest results to brief the council. "
        "Read available CSV/report files, extract WR, EV/trade, MDD, Sharpe. "
        "State facts only — no opinion. English, 5 bullet points max."
    )
    radi_user = (
        f"Agenda: {agenda[:300]}\n\n"
        "Scan the most recent backtest output files (reports/, project_output/) "
        "and summarize the key performance numbers: WR, EV, MDD, trade count, Gate 1/2 status. "
        "If no backtest files exist, state that clearly."
    )
    _du("Radi", status="SCANNING", task="Pre-council backtest scan")
    radi_scan = call_claude_cli(
        f"{radi_sys}\n\n{radi_user}", timeout=180
    )
    _dc("Radi", f"[SCAN] {radi_scan}")
    _dl("Pre-council: Radi backtest scan done")

    # Viktor — Gate 상태 및 수학적 현황 감사
    viktor_sys = (
        f"{_PERSONAS['Viktor']}\n\n{_PROJ_CTX}\n\n"
        "ROLE: You are auditing the mathematical state of the system before council. "
        "Check validation gate files, checkpoint status, and feature pipeline integrity. "
        "Report facts with numbers. English, 5 bullet points max."
    )
    viktor_user = (
        f"Agenda: {agenda[:300]}\n\n"
        "Check: (1) backtesting/validation.py Gate 1/2 thresholds, "
        "(2) latest checkpoint file in checkpoints/, "
        "(3) any existing validation results or r-multiple CSVs. "
        "Report what exists and what is missing."
    )
    _du("Viktor", status="AUDITING", task="Pre-council gate audit")
    viktor_scan = call_claude_cli(
        f"{viktor_sys}\n\n{viktor_user}", timeout=180
    )
    _dc("Viktor", f"[AUDIT] {viktor_scan}")
    _dl("Pre-council: Viktor gate audit done")

    key_facts = f"BACKTEST STATE (Radi):\n{radi_scan}\n\nGATE AUDIT (Viktor):\n{viktor_scan}"
    _save("pre_council_results_scan", "Council", "PreCouncil",
          f"# Pre-Council Results Scan\n\n## Backtest State (Radi)\n{radi_scan}"
          f"\n\n## Gate Audit (Viktor)\n{viktor_scan}")

    if _CP:
        _CP.mark("adversarial_debate", adversarial_disputes=key_facts[:300])
    return {"key_disputes": key_facts, "radi_scan": radi_scan, "viktor_scan": viktor_scan}


def run_grand_council(agenda: str, problem: str,
                      adv_result: dict = None, memory_ctx: str = "") -> str:
    """
    Viktor-led structured council.
    Opening: Viktor → Radi → Casandra (position statements).
    Each round: 6 challenge+rebuttal pairs = 12 exchanges.
    MAX_ROUNDS=3 → total 3+36=39 API calls.
    adv_result: output of run_adversarial_debate() — key_disputes injected into every call.
    memory_ctx: similar past sessions from SessionMemory — injected into system prompts.
    """
    global _CODE_DIGEST, _ADV_CONTEXT, _MEMORY_CONTEXT
    MAX_ROUNDS = 0  # Debate rounds disabled — opening statements + Demis synthesis only
    SEP = "─" * 70

    # Build code digest once — injected into every council call
    _dl("Building codebase digest for council...")
    _CODE_DIGEST = _build_code_digest()
    digest_lines = _CODE_DIGEST.count("\n")
    _dl(f"Code digest ready: {len(_CODE_DIGEST)} chars, {digest_lines} lines from {len(_COUNCIL_FILES)} files")

    # Adversarial debate context — injected into all council prompts
    _ADV_CONTEXT = ""
    if adv_result and adv_result.get("key_disputes"):
        _ADV_CONTEXT = (
            "\n\nPRE-COUNCIL ADVERSARIAL DEBATE — UNRESOLVED DISPUTES:\n"
            f"{adv_result['key_disputes']}\n"
            "These disputes are unresolved. Your council MUST address each one explicitly."
        )

    # Memory context — past similar sessions
    _MEMORY_CONTEXT = f"\n\n{memory_ctx}" if memory_ctx else ""

    agenda_lines = [l.strip() for l in agenda.split("\n") if l.strip()]
    agenda_summary = " | ".join(agenda_lines[:4])[:300]

    if _DASH:
        _DASH.council_mode = True
        _DASH.update("", phase="Phase 1 · Opening")

    history: list[tuple[str, str]] = []
    decisions: list[str] = []

    _dc("SYSTEM", f"[Council] {agenda_summary}")

    # ── Opening: Viktor → Radi → Casandra ─────────────────
    _dl("Opening statements — Viktor, Radi, Casandra")
    for lead in COUNCIL_ORDER:
        _du(lead["name"], status="SPEAKING", task="Opening statement")
        reply = _opening_statement(lead, agenda, problem)
        history.append((lead["name"], reply))
        _dc(lead["name"], reply)
        _dl(f"[Opening] {lead['name']} done")

    # ── Rounds: attack/rebuttal pairs ─────────────────────
    for round_n in range(1, MAX_ROUNDS + 1):
        _DASH and _DASH.update("", phase=f"Phase 1 · Round {round_n}/{MAX_ROUNDS} · Debate")
        _dl(f"Round {round_n}/{MAX_ROUNDS} — 6 attack/rebuttal pairs")

        for attacker, defender in ATTACK_PAIRS:
            # find defender's most recent statement
            defender_last = next(
                (msg for spk, msg in reversed(history) if spk == defender["name"]), ""
            )
            if not defender_last:
                continue

            # challenge
            _du(attacker["name"], status="SPEAKING",
                task=f"R{round_n} vs {defender['name']}")
            challenge_msg = _challenge_statement(attacker, defender, defender_last, round_n)
            tag = f"[>{defender['name']}]"
            history.append((attacker["name"], f"{tag} {challenge_msg}"))
            _dc(attacker["name"], f"{tag} {challenge_msg}")
            _dl(f"[R{round_n}] {attacker['name']} -> {defender['name']}")

            # rebuttal
            _du(defender["name"], status="SPEAKING",
                task=f"R{round_n} rebutting {attacker['name']}")
            rebuttal_msg = _rebuttal_statement(defender, attacker, challenge_msg, round_n)
            tag2 = f"[<{attacker['name']}]"
            history.append((defender["name"], f"{tag2} {rebuttal_msg}"))
            _dc(defender["name"], f"{tag2} {rebuttal_msg}")
            _dl(f"[R{round_n}] {defender['name']} rebuts {attacker['name']}")

        decisions = _extract_decisions(history, decisions)
        if decisions:
            _dl(f"{len(decisions)} decisions logged after round {round_n}")

    # ── User final input ───────────────────────────────────
    _DASH and _DASH.pause()

    print(f"\n{'═'*70}")
    print(f"  Council complete — {MAX_ROUNDS} rounds x 12 exchanges")
    if decisions:
        print(f"\n  Agreed decisions ({len(decisions)}):")
        for d in decisions:
            print(f"    + {d}")
    print(f"\n{SEP}")
    print("  Add a final comment, or Enter → 'conclude' to close.")
    print(SEP)

    while True:
        user_input = input("\n  YOU (or 'conclude'): ").strip()
        if user_input.lower() in ("conclude", "done", "종료", "끝", ""):
            if user_input and user_input.lower() not in ("conclude", "done", "종료", "끝"):
                history.append(("YOU", user_input)); _dc("YOU", user_input)
            break
        history.append(("YOU", user_input)); _dc("YOU", user_input)
        _DASH and _DASH.resume()
        # user comment → one final response round from each lead
        for lead in COUNCIL_ORDER:
            system = (
                f"{_PERSONAS[lead['name']]}\n\n{_PROJ_CTX}\n\n{_COUNCIL_RULES}"
            )
            recent = "\n".join(
                f"[{spk}]: {msg[:300]}" for spk, msg in history[-6:]
            )
            user_msg = (
                f"The project owner just said: \"{user_input}\"\n\n"
                f"Recent exchange:\n{recent}\n\n"
                f"{lead['name']}, respond to the owner's comment directly."
            )
            reply = _council_call(lead, system, user_msg)
            history.append((lead["name"], reply)); _dc(lead["name"], reply)
        _DASH and _DASH.pause()

    _dc("SYSTEM", f"[Council closed — {MAX_ROUNDS} rounds, {len(decisions)} decisions]")
    _dl("Council closed → Demis synthesis")
    _DASH and _DASH.resume()

    transcript = "\n\n".join(f"[{spk}]\n{msg}" for spk, msg in history)
    if decisions:
        transcript += "\n\n[AGREED DECISIONS]\n" + "\n".join(f"• {d}" for d in decisions)

    date_str = datetime.now().strftime("%Y-%m-%d")
    existing = list(PROJECT_FOLDER.glob(f"{date_str}-*_Result.md"))
    meeting_no = len(existing) + 1
    result_path = PROJECT_FOLDER / f"{date_str}-{meeting_no:02d}_Result.md"
    result_path.write_text(transcript, encoding="utf-8")
    _dl(f"saved → {result_path.name}")

    if _DASH: _DASH.council_mode = False
    if _CP: _CP.mark("council", transcript=transcript)
    return transcript


# ─────────────────────────────────────────────────────────────
# Phase 2: Demis — 합성 & 태스크 배분
# ─────────────────────────────────────────────────────────────
def demis_synthesize(transcript: str, memory_ctx: str = "") -> tuple[str, dict]:
    _DASH and _DASH.update("", phase="Phase 2 · Demis Synthesis")
    _du("Demis", task="회의 합성 & 태스크 배분")
    _dl("Phase 2 — Demis synthesizing plan")

    memory_section = f"\n\n{memory_ctx}" if memory_ctx else ""
    system = (
        "You are Demis (CEO) of a quantum crypto trading AI team.\n"
        "English only. Plain text, no markdown.\n"
        "Based on the Grand Council transcript, output exactly this structure:\n\n"
        "STRATEGY PLAN\n"
        "<concise plan, max 200 words>\n\n"
        "Radi's Tasks (Team Alpha)\n"
        "1. <concrete task: file path + function name + expected output>\n"
        "2. ...\n\n"
        "Casandra's Tasks (Team Beta)\n"
        "1. <concrete task>\n"
        "2. ...\n\n"
        "Viktor's Tasks (CTO)\n"
        "1. <concrete task>\n"
        "2. ...\n\n"
        "4-6 tasks per lead. Be specific. No filler."
        + memory_section
    )
    user = f"TRANSCRIPT:\n{transcript}\n\nSynthesize and assign."
    result = call_claude(DEMIS, system, user, timeout=180, tier="demis_synthesis")
    _save("demis_strategy_plan", "Demis", "Executive", result)
    _dl("Phase 2 complete")
    parsed = {
        "Radi":     _parse_tasks(result, "Radi"),
        "Casandra": _parse_tasks(result, "Casandra"),
        "Viktor":   _parse_tasks(result, "Viktor"),
    }
    if _CP: _CP.mark("synthesis", plan=result, task_map=parsed)
    return result, parsed


def _parse_tasks(text: str, lead: str) -> list[str]:
    """
    Extract numbered task list for a given lead from Demis's plan text.
    Handles both markdown (## Radi's Tasks) and plain-text (Radi's Tasks) formats.
    Joins continuation lines (indented lines after a numbered item).
    """
    # Find the section header and capture everything until the next Tasks section or end
    m = re.search(
        rf"#*\s*{re.escape(lead)}['\u2019]?s?\s+Tasks.*?\n(.*?)(?=\n\s*\w+['\u2019]?s?\s+Tasks|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if not m:
        return []

    section = m.group(1)
    tasks: list[str] = []
    current: str | None = None

    for line in section.splitlines():
        numbered = re.match(r'^\s*\d+\.\s+(.+)', line)
        if numbered:
            if current is not None:
                tasks.append(current)
            current = numbered.group(1).strip()
        elif current is not None and line.strip():
            # continuation line — append to current task
            current += " " + line.strip()

    if current is not None:
        tasks.append(current)

    return [t for t in tasks if t]


# ─────────────────────────────────────────────────────────────
# Phase 3/4: 팀 스프린트 (codex 구현 + claude 검토)
# ─────────────────────────────────────────────────────────────
MAX_REVISIONS = 2  # 팀장의 최대 수정 요청 횟수


def _lead_assign_tasks(lead: dict, members: list[dict], plan: str, lead_tasks: list[str]) -> dict[str, str]:
    """팀장 → 팀원별 구체적 태스크 할당.
    Lead assigns tasks based on member roles. Falls back to round-robin if
    parsing fails or lead times out.
    """
    member_roles = "\n".join(f"- {m['name']}: {m['role']}" for m in members)
    tasks_str    = "\n".join(f"{i+1}. {t[:200]}" for i, t in enumerate(lead_tasks)) if lead_tasks else "No tasks."
    sys_p = (
        f"You are {lead['name']}, team lead of {lead['dept']}.\n"
        "Assign each task below to the most suitable team member based on their role.\n"
        "Rules: each member gets at most one primary task; tasks with no match go to the most general member.\n"
        "Output format — one line per assignment, nothing else:\n"
        "[MEMBER_NAME]: <full task text>"
    )
    usr_p = (
        f"TEAM MEMBERS AND ROLES:\n{member_roles}\n\n"
        f"TASKS TO ASSIGN:\n{tasks_str}\n\n"
        "Assign now. Output only [NAME]: task lines."
    )
    _du(lead["name"], task="팀원 태스크 할당 중")
    kickoff = call_claude(lead, sys_p, usr_p, allow_tools=False, timeout=300,
                          tier="lead_assign")
    _save(f"kickoff_{lead['dept'].replace(' ','_')}", lead["name"], lead["dept"], kickoff)

    # ── Parse [NAME]: task assignments ──
    assignments: dict[str, str] = {}
    for m in members:
        pat = rf"\[{re.escape(m['name'])}\]\s*:?\s*(.+?)(?=\n\[|\Z)"
        match = re.search(pat, kickoff, re.DOTALL | re.IGNORECASE)
        if match:
            assignments[m["name"]] = match.group(1).strip()

    # ── Fallback: round-robin from lead_tasks if parsing missed any member ──
    for i, m in enumerate(members):
        if m["name"] not in assignments:
            assignments[m["name"]] = lead_tasks[i % len(lead_tasks)] if lead_tasks else "Implement your part of the plan."

    return assignments, kickoff


def _member_implement(member: dict, task: str, plan: str, kickoff: str,
                      feedback: str = "") -> str:
    """팀원 → claude -p (allow_tools=True) 구현.
    codex는 사용량 한도 문제로 제거. claude opus로 직접 구현.
    """
    revision_note = f"\n\nREVISION FEEDBACK from lead:\n{feedback}" if feedback else ""

    # ── Memory: inject this member's past task history ────────
    _hist = get_agent_history(member["name"], limit=5)
    history_block = format_agent_history(_hist) if _hist else ""

    sys_p = (
        f"You are {member['name']}, {member['role']}.\n"
        f"Working directory: {WORKSPACE}\n\n"
        "You have access to tools: Read, Edit, Write, Bash, and Google Search.\n\n"
        "MANDATORY: Use the ReAct pattern for ALL actions:\n"
        "  Thought: [Analyze — what do you know, what do you need?]\n"
        "  Action:  [Tool call or code change — be specific with file paths]\n"
        "  Observation: [Result of the action]\n"
        "  ... (repeat Thought/Action/Observation until task is complete)\n"
        "  Final Answer:\n"
        "    Files changed: [list]\n"
        "    Key decisions: [list]\n"
        "    Blockers: [list or 'none']\n"
        "    Papers cited: [Author (Year). Title. Venue. arxiv:ID]\n\n"
        "When searching for papers:\n"
        "  Thought: I need latest research on <topic> for this task\n"
        "  Action: Google Search('arxiv <topic> quant finance 2024 2025 2026')\n"
        "  Observation: [search results]\n"
    )
    # ── Research context (parallel, non-blocking) ───────────────
    _research_ctx = [""]
    def _fetch_research():
        try:
            role_lower = member.get("role", "").lower()
            if any(k in role_lower for k in ("quant", "ml", "risk", "backtest", "strategy", "quantum")):
                fields = ["qfin", "math", "cs"]
            elif "data" in role_lower:
                fields = ["cs", "qfin"]
            else:
                fields = ["qfin", "cs"]
            papers = search_papers(task, fields=fields, max_total=5)
            _research_ctx[0] = format_papers_for_prompt(papers, max_papers=4)
        except Exception:
            pass
    _rt = threading.Thread(target=_fetch_research, daemon=True)
    _rt.start()
    _rt.join(timeout=12)   # 12s 안에 못 가져오면 없이 진행

    research_block = (f"\n\n## Relevant Research Papers\n{_research_ctx[0]}"
                      if _research_ctx[0] else "")

    usr_p = (
        f"## Strategy Plan (summary)\n{plan[:600]}\n\n"
        f"## Lead Assignment\n{kickoff[:400]}\n\n"
        f"## Your Task\n{task}{revision_note}"
        f"{research_block}\n\n"
        + (f"{history_block}\n" if history_block else "")
        + "Implement now."
    )
    agent = {
        "name": member["name"],
        "model": member.get("model", MODEL_OPUS),
        "codex": member["codex"],
        "role": member["role"],
        "dept": member["dept"],
    }
    _du(member["name"], task=task[:50])

    # Part A: 분석/코드 작성 → Gemini 2.5 Pro (ReAct loop for file access)
    # Part B: 실행/검증/백테스트 → Claude Code CLI (실제 Bash/파일 도구)
    if _is_execution_task(task):
        _dl(f"[EXEC] {member['name']} — execution task → Claude Code CLI")
        return call_claude_exec(agent, sys_p, usr_p, timeout=600)

    # ReAct loop: model can Read/Write/Bash files iteratively
    full_prompt = (f"{sys_p}\n\n---\n\n{usr_p}" if sys_p else usr_p)
    return call_gemini_react(agent, full_prompt,
                             model_name=member.get("model", MODEL_OPUS),
                             timeout=600)


def _lead_review(lead: dict, member: dict, task: str,
                 impl_result: str) -> tuple[bool, str]:
    """팀장 → 팀원 결과 검토. 반환: (approved, feedback)"""
    _du(lead["name"], status="REVIEW", task=f"Reviewing {member['name']}'s work")
    sys_p = (
        f"You are {lead['name']}, team lead.\n"
        "Review the team member's implementation result.\n"
        "If the task is COMPLETE and correct, respond with exactly: APPROVED\n"
        "If revision is needed, respond with: REVISION: <specific feedback on what to fix>\n"
        "Be concise and precise."
    )
    usr_p = (
        f"TASK ASSIGNED:\n{task}\n\n"
        f"MEMBER: {member['name']}\n"
        f"IMPLEMENTATION RESULT:\n{impl_result[:2000]}\n\n"
        "Is this task complete? Respond APPROVED or REVISION: <feedback>"
    )
    review = call_claude(lead, sys_p, usr_p, timeout=120, tier="lead_review")
    if "TIMEOUT" in review or "ERROR" in review:
        return True, ""  # lead unavailable — treat as approved, move on
    if review.strip().upper().startswith("APPROVED"):
        return True, ""
    feedback = review[len("REVISION:"):].strip() if review.upper().startswith("REVISION:") else review
    return False, feedback


def extract_sprint_decision(member_name: str, task: str, result: str) -> dict:
    """
    팀원 스프린트 결과에서 구조화된 결정 추출.
    TradingAgents SignalProcessor 패턴 적용.
    Returns dict with: action, confidence, files_changed, gate_impact, blockers, papers_cited
    """
    extract_agent = {"name": member_name, "model": MODEL_TIER["signal_extract"],
                     "codex": "EX", "role": "extractor", "dept": "System"}
    prompt = (
        "Extract a structured decision from this sprint result. "
        "Return ONLY a valid JSON object with exactly these keys:\n"
        '{"action": "IMPLEMENTED|BLOCKED|PARTIAL|DEFERRED", '
        '"confidence": 0-100, '
        '"files_changed": ["file1", "file2"], '
        '"gate_impact": "PASS|FAIL|PENDING|NA", '
        '"blockers": ["blocker1"], '
        '"papers_cited": ["Author (Year) Title"]}\n\n'
        f"MEMBER: {member_name}\n"
        f"TASK: {task[:200]}\n"
        f"RESULT:\n{result[:1200]}"
    )
    raw = call_gemini(extract_agent, prompt, MODEL_TIER["signal_extract"], 60)
    try:
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"action": "UNKNOWN", "confidence": 0, "files_changed": [],
            "gate_impact": "NA", "blockers": [], "papers_cited": []}


def run_team_sprint(lead: dict, members: list[dict],
                    plan: str, lead_tasks: list[str]) -> str:
    _DASH and _DASH.update("", phase=f"Sprint · {lead['dept']}")
    _dl(f"{lead['dept']} sprint starting")

    # 태스크 할당
    assignments, kickoff = _lead_assign_tasks(lead, members, plan, lead_tasks)

    # 각 팀원: 병렬 구현 → 팀장 검토 → (필요시 수정) 루프
    final_results:   dict[str, str]  = {}
    final_decisions: dict[str, dict] = {}
    cp_lock = threading.Lock()

    def _run_member(member: dict) -> None:
        task = assignments.get(member["name"], "Implement assigned task.")

        # ── Checkpoint: skip member if already completed in a previous run ──
        if _CP:
            cached = _CP.get_member(lead["name"], member["name"])
            if cached:
                _dl(f"{member['name']} — resumed from checkpoint (skipping)")
                _du(member["name"], status="DONE", task=task[:50], msg="[checkpoint resume]")
                final_results[member["name"]] = cached
                return

        result   = ""
        feedback = ""
        approved = False

        for attempt in range(1 + MAX_REVISIONS):
            round_label = "initial" if attempt == 0 else f"revision {attempt}"
            _dl(f"{member['name']} — {round_label} implementation")
            result = _member_implement(member, task, plan, kickoff, feedback)

            _save(
                f"{member['name'].lower()}_{round_label.replace(' ','_')}",
                member["name"], member["dept"], result
            )

            approved, feedback = _lead_review(lead, member, task, result)
            if approved:
                _dl(f"{lead['name']} APPROVED {member['name']}'s work")
                break
            else:
                _dl(f"{lead['name']} requested revision from {member['name']}: {feedback[:60]}")

        if not approved:
            _dl(f"{member['name']} max revisions reached — using last result")

        with cp_lock:
            # Only cache real results — skip [NO RESPONSE] / TIMEOUT / ERROR
            if _CP and result and not any(x in result for x in ("[NO RESPONSE]", "TIMEOUT", "ERROR")):
                _CP.set_member(lead["name"], member["name"], result)
            final_results[member["name"]] = result
            # Extract structured decision (Flash — fast)
            decision = extract_sprint_decision(member["name"], task, result)
            final_decisions[member["name"]] = decision

        # ── Memory: persist member task + result (non-blocking) ──
        if result and not any(x in result for x in ("[NO RESPONSE]", "TIMEOUT", "ERROR")):
            _sid = _CP.data.get("session_id", "unknown") if _CP else "unknown"
            _result_summary = result[:300]
            threading.Thread(
                target=save_agent_memory,
                args=(member["name"], task, _result_summary, _sid),
                daemon=True,
            ).start()
            threading.Thread(
                target=update_task_status,
                args=(_sid, member["name"], task, "DONE", _result_summary[:200]),
                daemon=True,
            ).start()

    # Launch all members in parallel (all use Gemini 2.5 Pro)
    threads = [threading.Thread(target=_run_member, args=(m,), daemon=True)
               for m in members]
    for i, t in enumerate(threads):
        t.start()
        if i < len(threads) - 1:
            time.sleep(0.5)

    for t in threads: t.join()

    # 팀장 → 전체 결과 취합 보고서
    sprint_log = "\n\n".join(
        f"[{m['name']}]\n{final_results.get(m['name'],'[NO RESULT]')}"
        for m in members
    )
    # Structured decision table
    decision_table = (
        "| Member | Action | Confidence | Gate | Files Changed |\n"
        "|--------|--------|-----------|------|---------------|\n"
    )
    for m in members:
        d = final_decisions.get(m["name"], {})
        files_str = ", ".join(d.get("files_changed", [])[:2]) or "-"
        decision_table += (
            f"| {m['name']} | {d.get('action','?')} | {d.get('confidence','?')}% | "
            f"{d.get('gate_impact','?')} | {files_str} |\n"
        )

    sum_sys = (
        f"You are {lead['name']}, team lead of {lead['dept']}.\n"
        "Compile a Sprint Completion Report (200 words):\n"
        "1. What each member implemented\n"
        "2. Key decisions made\n"
        "3. Remaining blockers or follow-ups\n"
        "This report will be sent to Demis (CEO)."
    )
    sum_usr = f"SPRINT LOG:\n{sprint_log[:4000]}\n\nWrite the Sprint Completion Report."
    _du(lead["name"], task="스프린트 결과 취합 보고서 작성")
    summary = call_claude(lead, sum_sys, sum_usr, allow_tools=False,
                          tier="sprint_summary")

    _save(f"sprint_completion_{lead['dept'].replace(' ','_')}", lead["name"], lead["dept"],
          f"# {lead['dept']} Sprint Report\n**Date:** {datetime.now()}\n\n"
          f"## Summary\n{summary}\n\n"
          f"## Structured Decisions\n{decision_table}\n\n"
          f"## Full Log\n\n{sprint_log}")
    _dl(f"{lead['dept']} sprint complete")
    # Save sprint summary to checkpoint
    if _CP:
        sprint_key = lead["name"].lower()[:5]
        _CP.data["summaries"][sprint_key if sprint_key in _CP.data["summaries"]
                              else lead["codex"].lower()] = summary
        phase_key = {"Radi": "alpha_sprint", "Casandra": "beta_sprint"}.get(lead["name"], "alpha_sprint")
        _CP.mark(phase_key)
    return summary


# ─────────────────────────────────────────────────────────────
# Phase 2.5: Plan Review — user approves before sprints fire
# ─────────────────────────────────────────────────────────────
def review_plan(plan: str, task_map: dict) -> dict:
    """
    Pause after Demis synthesis and show the plan + task assignments.
    User can:
      - Enter       → approve all, proceed
      - skip alpha  → skip Team Alpha sprint
      - skip beta   → skip Team Beta sprint
      - skip cto    → skip Viktor solo sprint
      - edit <lead> → paste replacement task list for that lead (multiline, blank line = done)
      - abort       → cancel all sprints, jump to final report with no sprint data
    Returns the (possibly modified) task_map, or None if aborted.
    """
    if _DASH: _DASH.pause()

    SEP  = "═" * 70
    SEP2 = "─" * 70

    print(f"\n{SEP}")
    print("  PHASE 2 COMPLETE — DEMIS STRATEGY PLAN")
    print(SEP)

    # Show plan (first 60 lines)
    plan_lines = plan.splitlines()
    for l in plan_lines[:60]:
        print(f"  {l}")
    if len(plan_lines) > 60:
        print(f"  … ({len(plan_lines) - 60} more lines — see project_output/)")
    print(SEP2)

    # Show parsed task assignments
    for lead_name, tasks in task_map.items():
        if tasks:
            print(f"\n  {lead_name}'s Tasks:")
            for i, t in enumerate(tasks, 1):
                print(f"    {i}. {t}")
        else:
            print(f"\n  {lead_name}: (no tasks parsed)")
    print(f"\n{SEP}")
    print("  OPTIONS:")
    print("    Enter          → approve all, start sprints")
    print("    skip alpha     → skip Team Alpha sprint")
    print("    skip beta      → skip Team Beta sprint")
    print("    skip cto       → skip Viktor CTO sprint")
    print("    edit Radi      → replace Radi's task list (paste, blank line = done)")
    print("    edit Casandra  → replace Casandra's task list")
    print("    edit Viktor    → replace Viktor's task list")
    print("    abort          → skip all sprints, go straight to final report")
    print(SEP)

    skipped: set[str] = set()

    while True:
        cmd = input("\n  Plan review: ").strip().lower()

        if cmd in ("", "approve", "y", "yes", "go"):
            break

        elif cmd == "abort":
            print("  [Sprints aborted — final report will be summary-only]")
            if _DASH: _DASH.resume()
            return None

        elif cmd.startswith("skip "):
            target = cmd[5:].strip()
            if target in ("alpha", "radi"):
                skipped.add("Radi"); print("  → Team Alpha sprint will be skipped")
            elif target in ("beta", "casandra"):
                skipped.add("Casandra"); print("  → Team Beta sprint will be skipped")
            elif target in ("cto", "viktor"):
                skipped.add("Viktor"); print("  → Viktor CTO sprint will be skipped")
            else:
                print(f"  Unknown team '{target}'. Try: skip alpha / skip beta / skip cto")

        elif cmd.startswith("edit "):
            lead_name = cmd[5:].strip().capitalize()
            if lead_name not in task_map:
                # try partial match
                matches = [k for k in task_map if k.lower().startswith(cmd[5:].strip().lower())]
                if matches:
                    lead_name = matches[0]
                else:
                    print(f"  Unknown lead. Use: edit Radi | edit Casandra | edit Viktor")
                    continue
            print(f"  Enter new tasks for {lead_name} (one per line, blank line = done):")
            new_tasks = []
            while True:
                line = input(f"  [{lead_name}] > ").strip()
                if not line:
                    break
                new_tasks.append(line)
            if new_tasks:
                task_map[lead_name] = new_tasks
                print(f"  → {lead_name}'s tasks updated ({len(new_tasks)} tasks)")
            else:
                print(f"  → No changes (empty input)")

        else:
            print("  Unknown command. Enter / skip alpha / skip beta / skip cto / edit <name> / abort")

    # Apply skips by clearing task lists and marking
    for lead_name in skipped:
        task_map[lead_name] = None   # None = skip signal

    if _CP: _CP.mark("plan_review", approved_map=task_map)
    if _DASH: _DASH.resume()
    return task_map


# ─────────────────────────────────────────────────────────────
# Phase 4.5: Viktor solo sprint (CTO research & validation)
# ─────────────────────────────────────────────────────────────
def run_viktor_solo(plan: str, viktor_tasks: list[str]) -> str:
    _DASH and _DASH.update("", phase="Phase 4.5 · Viktor CTO Solo Sprint")
    _du("Viktor", task="OOS validation & mathematical analysis")
    _dl("Phase 4.5 — Viktor solo validation sprint")

    tasks_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(viktor_tasks)) if viktor_tasks else "See plan."
    sys_p = (
        "You are Viktor, Quant Researcher CTO.\n\n"
        "You have FULL READ ACCESS to all project files via tools:\n"
        "  - glob_files('project_output/*.csv')  → discover available output files\n"
        "  - glob_files('project_output/*.md')   → discover reports and logs\n"
        "  - read_file('project_output/<file>')  → read any file you need\n"
        "  - read_file('src/...')                → read source code\n"
        "  - bash('python scripts/...')          → run validation scripts directly\n\n"
        "MANDATORY FIRST STEP: Use glob_files + read_file to read ALL relevant output files\n"
        "before writing your report. Do not rely only on what is given in the prompt.\n\n"
        "Use the ReAct pattern:\n"
        "  Thought: [What files/data do I need to read?]\n"
        "  Action:  [glob_files / read_file / bash]\n"
        "  Observation: [File contents / results]\n"
        "  ... (repeat until you have read all relevant data)\n"
        "  Final Answer: [Full CTO Validation Report]\n\n"
        "DELIVERABLE CREATION (MANDATORY):\n"
        "For each task assigned to you, CREATE the actual file on disk using the Write tool.\n"
        "Example: if task says 'write docs/validation_protocols.md', use Write tool to create that file.\n"
        "Do NOT just describe what would be in the file — actually write it to disk.\n\n"
        "Final Answer must contain:\n"
        "1. OOS Gate Criteria (Gate 1 + Gate 2 — with actual numbers from the files)\n"
        "2. Statistical Validity Review (based on ACTUAL backtest output you read)\n"
        "3. Regime Theory Analysis (when does the feature set fail?)\n"
        "4. Latest Research (>=3 papers cited)\n"
        "5. Recommended Implementation Sequence\n"
        "6. Fallback Plan\n"
        "7. List of files created (paths confirmed by Write tool)\n"
        "English only. Mathematically precise. Cite actual file names you read."
    )
    # ── Pre-fetch research for Viktor (math + qfin + physics) ──
    _viktor_research = [""]
    def _fetch_viktor_research():
        try:
            papers = search_papers(tasks_str, fields=["math", "qfin", "physics", "cs"],
                                   max_total=8)
            _viktor_research[0] = format_papers_for_prompt(papers, max_papers=6)
        except Exception:
            pass
    _vrt = threading.Thread(target=_fetch_viktor_research, daemon=True)
    _vrt.start()
    _vrt.join(timeout=15)
    research_block = (f"\n\n## Pre-fetched Research (arXiv + OpenAlex + Crossref)\n"
                      f"{_viktor_research[0]}" if _viktor_research[0] else "")

    usr_p = (
        f"STRATEGY PLAN:\n{plan[:1500]}\n\n"
        f"YOUR ASSIGNED TASKS:\n{tasks_str}"
        f"{research_block}\n\n"
        "Start by exploring project_output/ with glob_files, then read the files you need.\n"
        "Write the CTO Validation Report now."
    )
    report = call_claude_exec(VIKTOR, sys_p, usr_p, timeout=480)
    _save("cto_validation_report", "Viktor", "CTO", report)
    _du("Viktor", status="DONE", task="CTO validation report complete")
    _dl("Phase 4.5 complete")
    if _CP:
        _CP.data["summaries"]["cto"] = report
        _CP.mark("cto_sprint")
    return report


# ─────────────────────────────────────────────────────────────
# Phase 4.7: Viktor Risk Audit (replaces 3-perspective debate)
# ─────────────────────────────────────────────────────────────
def run_risk_debate(plan: str, alpha_sum: str, beta_sum: str, cto_sum: str) -> str:
    """
    스프린트 완료 후 Viktor 단독 수학적 리스크 감사.
    토론 대신 실제 백테스트 숫자 기반 Kelly fraction + Gate 판정.
    1 API call (debate 6 → 1로 축소)
    """
    _DASH and _DASH.update("", phase="Phase 4.7 · Risk Audit")
    _dl("Post-sprint Viktor Risk Audit — data-driven Kelly + Gate verdict")

    sprint_ctx = (
        f"ALPHA SPRINT RESULTS:\n{alpha_sum[:500]}\n\n"
        f"BETA SPRINT RESULTS:\n{beta_sum[:500]}\n\n"
        f"CTO SPRINT RESULTS:\n{cto_sum[:500]}"
    )

    viktor_sys = (
        f"{_PERSONAS['Viktor']}\n\n{_PROJ_CTX}\n\n"
        "ROLE: Post-sprint mathematical risk auditor. Read actual backtest output files, "
        "compute Kelly fraction, verify Gate 1/2 status. No debate — only numbers and verdicts."
    )
    viktor_user = (
        f"Sprint results summary:\n{sprint_ctx}\n\n"
        "Using actual files in reports/ and project_output/, perform the risk audit:\n"
        "1. Read the latest r-multiple CSV. Compute: WR, avg_W, avg_L, EV/trade.\n"
        "2. Gate 1: WR > 25.4% (BEP)? PASS/FAIL.\n"
        "3. Gate 2: Bootstrap p-value < 0.05 on mean R-multiple > 0? PASS/FAIL.\n"
        "4. Kelly fraction: f* = (p/|SL| - q/|TP|) where p=WR, q=1-WR, TP=3×ATR, SL=1×ATR.\n"
        "5. VERDICT: INCREASE / MAINTAIN / REDUCE position sizing — one word + one sentence rationale.\n"
        "If no backtest files exist, state that clearly and output VERDICT: MAINTAIN (no data)."
    )
    _du("Viktor", status="AUDITING", task="Post-sprint risk audit")
    verdict = call_claude_cli(f"{viktor_sys}\n\n{viktor_user}", timeout=240)
    _dc("Viktor", f"[RISK-AUDIT] {verdict}")
    _dl("Risk audit complete")

    full_report = f"# Risk Audit Report\n\n## Viktor Mathematical Risk Audit\n{verdict}"
    _save("risk_audit_report", "Council", "RiskAudit", full_report)
    if _CP:
        _CP.mark("risk_debate", risk_verdict=verdict[:200])
    return full_report


# ─────────────────────────────────────────────────────────────
# Phase 5: Demis → 최종 보고서 (사용자에게)
# ─────────────────────────────────────────────────────────────
def demis_final_report(plan: str, alpha_summary: str, beta_summary: str,
                       cto_summary: str = "", risk_summary: str = "") -> str:
    _DASH and _DASH.update("", phase="Phase 5 · Demis Final Report")
    _du("Demis", task="최종 보고서 작성 중")
    _dl("Phase 5 — Demis writing final report")

    system = (
        "You are Demis (CEO). Write the FINAL REPORT to the project owner.\n\n"
        "You have FULL READ ACCESS to all project files via tools:\n"
        "  - glob_files('project_output/*.md')  → see all sprint reports and logs\n"
        "  - glob_files('project_output/*.csv') → see all backtest/data outputs\n"
        "  - read_file('project_output/<file>') → read any report or result\n"
        "  - read_file('src/...')               → read source code if needed\n\n"
        "MANDATORY: Before writing the report, use glob_files to discover what was produced\n"
        "this sprint, then read the key files (sprint completions, CTO report, risk debate).\n"
        "Your report must reference ACTUAL file contents, not just the summaries in the prompt.\n\n"
        "STRUCTURE:\n"
        "# Final Report — [Date]\n\n"
        "## Executive Summary (3 sentences)\n\n"
        "## What Was Accomplished\n"
        "- Team Alpha: ...\n"
        "- Team Beta: ...\n"
        "- CTO (Viktor): ...\n\n"
        "## Key Decisions Made\n\n"
        "## Risk Verdict (Viktor audit — Kelly fraction + Gate verdict)\n\n"
        "## Research & Papers Referenced\n\n"
        "## Next Steps (max 5 bullets)\n\n"
        "## Blockers / Risks\n\n"
        "---\n"
        "Report compiled by Demis (CEO)"
    )
    risk_section = f"\n\nRISK DEBATE VERDICT:\n{risk_summary[:800]}" if risk_summary else ""
    user = (
        f"STRATEGY PLAN:\n{plan[:1000]}\n\n"
        f"TEAM ALPHA SPRINT:\n{alpha_summary}\n\n"
        f"TEAM BETA SPRINT:\n{beta_summary}\n\n"
        f"CTO (VIKTOR) SPRINT:\n{cto_summary}\n\n"
        f"{risk_section}\n\n"
        f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "Start by exploring project_output/ with glob_files to see all sprint artifacts,\n"
        "then read the ones relevant to your report. Write the final report now."
    )
    report = call_claude_exec(DEMIS, system, user, timeout=360)
    _save("final_report", "Demis", "Executive", report)
    _dl("Final report complete")
    if _CP:
        _CP.mark("final_report")
        _CP.clear()   # session complete — remove checkpoint

    # ── Memory: persist completed session (non-blocking) ─────
    _sid = _CP.data.get("session_id", str(uuid.uuid4())) if _CP else str(uuid.uuid4())
    _agenda_snap  = _CP.data.get("agenda",  "")[:1000] if _CP else ""
    _disputes_snap = _CP.data.get("adversarial_disputes", "")[:500] if _CP else ""
    _risk_snap    = _CP.data.get("risk_verdict", "")[:100] if _CP else ""
    threading.Thread(
        target=save_session,
        args=(_sid, _agenda_snap, plan[:2000], report,
              {"alpha": alpha_summary[:300], "beta": beta_summary[:300], "cto": cto_summary[:300]},
              _disputes_snap, _risk_snap),
        daemon=True,
    ).start()
    github_push_async(_sid, report)

    return report


# ─────────────────────────────────────────────────────────────
# Sprint execution helper (shared by normal run + resume)
# ─────────────────────────────────────────────────────────────
def _run_sprints(plan: str, approved_map: dict) -> tuple[str, str, str, str]:
    """Execute all team sprints in parallel, then run risk debate. Returns (alpha, beta, cto, risk) summaries."""
    alpha_sum = beta_sum = cto_sum = "[SKIPPED]"

    if approved_map is None:
        _dl("All sprints aborted — proceeding to final report")
        return alpha_sum, beta_sum, cto_sum, "[SKIPPED]"

    results: dict[str, str] = {}
    sprint_lock = threading.Lock()

    def _sprint(key, fn, *args):
        r = fn(*args)
        with sprint_lock:
            results[key] = r

    threads = []
    if approved_map.get("Radi") is not None:
        threads.append(threading.Thread(target=_sprint, daemon=True,
            args=("alpha", run_team_sprint, RADI, ALPHA_MEMBERS, plan, approved_map.get("Radi", []))))
    else:
        _dl("Team Alpha sprint skipped")

    if approved_map.get("Casandra") is not None:
        threads.append(threading.Thread(target=_sprint, daemon=True,
            args=("beta", run_team_sprint, CASANDRA, BETA_MEMBERS, plan, approved_map.get("Casandra", []))))
    else:
        _dl("Team Beta sprint skipped")

    if approved_map.get("Viktor") is not None:
        threads.append(threading.Thread(target=_sprint, daemon=True,
            args=("cto", run_viktor_solo, plan, approved_map.get("Viktor", []))))
    else:
        _dl("Viktor CTO sprint skipped")

    for t in threads: t.start()
    for t in threads: t.join()

    alpha_sum = results.get("alpha", alpha_sum)
    beta_sum  = results.get("beta",  beta_sum)
    cto_sum   = results.get("cto",   cto_sum)

    # Phase 4.7: 3-perspective risk debate (after all sprints complete)
    if approved_map is not None and any(
        approved_map.get(k) is not None for k in ["Radi", "Casandra", "Viktor"]
    ):
        risk_sum = run_risk_debate(plan, alpha_sum, beta_sum, cto_sum)
    else:
        risk_sum = "[SKIPPED]"

    return alpha_sum, beta_sum, cto_sum, risk_sum


def _print_final(final: str):
    SEP = "═" * 70
    print(f"\n{SEP}")
    print("  FINAL REPORT FROM DEMIS (CEO)")
    print(SEP)
    print(final)
    print(f"\n{SEP}")
    print(f"  Artifacts → {PROJECT_FOLDER.absolute()}")
    print(SEP + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    global _DASH, _CP

    try:
        import rich  # noqa
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"], check=True)

    SEP = "═" * 70

    # ── 1) Detect agenda state (_load_md returns None for empty template) ───────
    agenda_text = _load_md(str(AGENDA_FILE)) or ""

    # ── 2) Check for incomplete checkpoint ────────────────────────────────────
    existing_cp = Checkpoint.load()
    if existing_cp and not existing_cp.is_complete:
        phases = existing_cp.data.get("phases_done", [])
        council_done = "council" in phases

        if council_done:
            # Checkpoint has useful state — offer resume
            d = existing_cp.data
            created = d.get("created_at", "?")[:19]
            agenda_preview = d.get("agenda", "")[:60]
            done_members = list(d.get("member_results", {}).keys())
            print(f"\n{SEP}")
            print("  INCOMPLETE SESSION — checkpoint has progress to resume")
            print(f"  Started   : {created}")
            print(f"  Agenda    : {agenda_preview}")
            print(f"  Phases    : {', '.join(phases)}")
            if done_members:
                print(f"  Members done: {', '.join(m.split(':')[1] for m in done_members)}")
            print(SEP)
            print("  r / resume  → continue from checkpoint")
            print("  n / new     → discard, start fresh")
            choice = input("  Choice [r/n]: ").strip().lower()
            if choice in ("r", "resume", ""):
                _CP = existing_cp
                _DASH = Dashboard()
                _DASH.register(ALL_AGENTS)
                _resume_from_checkpoint()
                return
            else:
                existing_cp.clear()
        else:
            # Council never completed → checkpoint is stale garbage, discard it
            existing_cp.clear()
            _dl("Stale checkpoint discarded (council was never completed)")

    # ── 3) Agenda empty → scan project_output for unfinished OR re-sprint ──────
    if not agenda_text:
        # Also find most recent plan even if final_report exists (for re-sprint)
        all_plans = sorted(PROJECT_FOLDER.glob("demis_strategy_plan-Demis-Executive-*.md"), reverse=True)
        session = _scan_last_session()

        if session or all_plans:
            plan_path = (session.get("plan_path") if session else None) or (all_plans[0] if all_plans else None)
            date_tag  = session["date_tag"] if session else (all_plans[0].stem[-10:] if all_plans else "?")
            done_members = list(session["member_results"].keys()) if session else []
            done_sprints = [p for p in (session["phases_done"] if session else []) if p.endswith("_sprint")]
            has_final = bool(list(PROJECT_FOLDER.glob(f"final_report-Demis-Executive-{date_tag}.md")))

            print(f"\n{SEP}")
            if session:
                print("  AGENDA IS EMPTY — unfinished work found in project_output/")
            else:
                print("  AGENDA IS EMPTY — previous session found (sprint results may need re-run)")
            print(f"  Session date   : {date_tag}")
            if plan_path:
                print(f"  Strategy plan  : {plan_path.name}")
            if done_members:
                print(f"  Members done   : {', '.join(m.split(':')[1] for m in done_members)}")
            if done_sprints:
                print(f"  Sprints done   : {', '.join(done_sprints)}")
            if has_final:
                print("  Final report   : exists (sprint results may have been empty)")
            print(SEP)
            print("  r / resume  → resume from last session (skip done work)")
            print("  s / re-sprint → re-run ALL sprints from saved plan (ignore previous results)")
            print("  n / new     → set new agenda and hold a fresh council")
            choice = input("  Choice [r/s/n]: ").strip().lower()
            if choice in ("r", "resume", ""):
                if session:
                    _run_from_last_result(session)
                    return
            elif choice in ("s", "re-sprint", "sprint"):
                if plan_path:
                    _do_resprint(plan_path)
                    return
                else:
                    print("  No strategy plan found — cannot re-sprint.")
                    return

    # ── 3) Check for re-sprint even when agenda has content ──────────────────
    all_plans = sorted(PROJECT_FOLDER.glob("demis_strategy_plan-Demis-Executive-*.md"), reverse=True)
    if all_plans:
        print(f"\n{SEP}")
        print(f"  Saved strategy plan found: {all_plans[0].name}")
        print("  s / re-sprint → skip council, re-run sprints from saved plan")
        print("  n / new       → run full council with current agenda")
        choice = input("  Choice [s/n]: ").strip().lower()
        if choice in ("s", "re-sprint", "sprint"):
            _do_resprint(all_plans[0])
            return

    # ── 4) Normal flow: council → synthesis → review → sprints → report ──────
    _CP = Checkpoint()
    agenda, problem = get_user_agenda()
    _CP.data.update({"agenda": agenda, "problem": problem})
    _CP.save()

    _DASH = Dashboard()
    _DASH.register(ALL_AGENTS)

    with _DASH:
        _dl("Orchestration initialized")

        # Phase 0: Retrieve similar past sessions for context
        memory_ctx = _MEM.retrieve_similar(agenda)
        if memory_ctx:
            _dl(f"Memory: {memory_ctx.count(chr(10))} lines of past session context loaded")

        # Phase 0.1: Augment with SQLite/ChromaDB cross-session memory
        try:
            _db_snippets = search_memory(agenda[:500], top_k=3)
            if _db_snippets:
                _db_block = "\n\n=== DEEP MEMORY (SQLite/ChromaDB past sessions) ===\n"
                _db_block += "\n---\n".join(_db_snippets[:3])
                memory_ctx = (memory_ctx + _db_block) if memory_ctx else _db_block
                _dl(f"Deep memory: {len(_db_snippets)} cross-session snippets loaded")
        except Exception:
            pass

        # Phase 0.5: Adversarial Pre-Council Debate (Radi↔Viktor)
        adv_result = run_adversarial_debate(agenda, problem)

        # Phase 1: Grand Council (with adv context + memory injected)
        transcript = run_grand_council(agenda, problem,
                                       adv_result=adv_result, memory_ctx=memory_ctx)

        # Phase 2: Demis synthesis (with memory context)
        plan, task_map = demis_synthesize(transcript, memory_ctx=memory_ctx)
        approved_map = review_plan(plan, task_map)

        # Phase 3/4: Team sprints + Phase 4.7: Risk debate
        alpha_sum, beta_sum, cto_sum, risk_sum = _run_sprints(plan, approved_map)

        # Phase 5: Final report (with risk verdict)
        final = demis_final_report(plan, alpha_sum, beta_sum, cto_sum, risk_sum)

        _DASH.update("", phase="COMPLETE")
        _dl("All phases complete")

    # Save session to memory for future runs
    decisions = []
    if _CP and _CP.data:
        adv_disputes = _CP.data.get("adversarial_disputes", "")
        risk_verdict = _CP.data.get("risk_verdict", "")
        if adv_disputes:
            decisions.append(f"Pre-debate disputes: {adv_disputes[:150]}")
        if risk_verdict:
            decisions.append(f"Risk verdict: {risk_verdict[:150]}")
    _MEM.store(
        session_id=_CP.data.get("session_id", str(uuid.uuid4())) if _CP else str(uuid.uuid4()),
        agenda=agenda,
        decisions=decisions,
        sprint_outcomes={
            "alpha": alpha_sum[:200] if alpha_sum else "",
            "beta":  beta_sum[:200]  if beta_sum  else "",
            "cto":   cto_sum[:200]   if cto_sum   else "",
            "risk":  risk_sum[:200]  if risk_sum  else "",
        }
    )

    _archive_agenda()
    _print_final(final)


def _resume_from_checkpoint():
    """Resume an interrupted session using the global _CP checkpoint."""
    global _DASH
    d = _CP.data
    plan      = d.get("plan", "")
    transcript= d.get("transcript", "")

    SEP = "═" * 70
    print(f"\n{SEP}")
    print("  RESUMING FROM CHECKPOINT")
    phases = d.get("phases_done", [])

    with _DASH:
        _dl("Resuming from checkpoint")

        # Redo synthesis if council done but synthesis not
        if "council" in phases and "synthesis" not in phases:
            _dl("Re-running Demis synthesis (interrupted before plan was saved)")
            plan, task_map = demis_synthesize(transcript)
        elif "synthesis" in phases:
            task_map = d.get("task_map", {})
            _dl(f"Synthesis restored from checkpoint ({len(task_map)} lead task lists)")
        else:
            # Council not even done — can't resume, must restart
            _DASH.pause()
            print(f"\n{SEP}")
            print("  Council was not completed. Cannot resume — please start fresh.")
            print(SEP)
            _CP.clear()
            return

        # Plan review if not done yet
        if "plan_review" not in phases:
            approved_map = review_plan(plan, task_map)
        else:
            approved_map = d.get("approved_map", task_map)
            _dl("Plan review restored from checkpoint")

        # Load any sprint summaries already saved
        sums = d.get("summaries", {})
        alpha_sum = sums.get("alpha", sums.get("radi", "[SKIPPED]")) or "[SKIPPED]"
        beta_sum  = sums.get("beta",  sums.get("casan", "[SKIPPED]")) or "[SKIPPED]"
        cto_sum   = sums.get("cto",   "[SKIPPED]") or "[SKIPPED]"

        # Run only incomplete sprints — all in parallel
        if approved_map is not None:
            resume_results: dict[str, str] = {}
            resume_lock = threading.Lock()

            def _rsprint(key, fn, *args):
                r = fn(*args)
                with resume_lock:
                    resume_results[key] = r

            rthreads = []
            if "alpha_sprint" not in phases and approved_map.get("Radi") is not None:
                rthreads.append(threading.Thread(target=_rsprint, daemon=True,
                    args=("alpha", run_team_sprint, RADI, ALPHA_MEMBERS, plan, approved_map.get("Radi", []))))
            elif "alpha_sprint" in phases:
                _dl("Team Alpha sprint already done (checkpoint)")

            if "beta_sprint" not in phases and approved_map.get("Casandra") is not None:
                rthreads.append(threading.Thread(target=_rsprint, daemon=True,
                    args=("beta", run_team_sprint, CASANDRA, BETA_MEMBERS, plan, approved_map.get("Casandra", []))))
            elif "beta_sprint" in phases:
                _dl("Team Beta sprint already done (checkpoint)")

            if "cto_sprint" not in phases and approved_map.get("Viktor") is not None:
                rthreads.append(threading.Thread(target=_rsprint, daemon=True,
                    args=("cto", run_viktor_solo, plan, approved_map.get("Viktor", []))))
            elif "cto_sprint" in phases:
                _dl("Viktor CTO sprint already done (checkpoint)")

            for t in rthreads: t.start()
            for t in rthreads: t.join()

            alpha_sum = resume_results.get("alpha", alpha_sum)
            beta_sum  = resume_results.get("beta",  beta_sum)
            cto_sum   = resume_results.get("cto",   cto_sum)

        # Risk debate if sprints ran
        if any(s not in ("[SKIPPED]", "") for s in [alpha_sum, beta_sum, cto_sum]):
            risk_sum = run_risk_debate(plan, alpha_sum, beta_sum, cto_sum)
        else:
            risk_sum = "[SKIPPED]"
        final = demis_final_report(plan, alpha_sum, beta_sum, cto_sum, risk_sum)
        _DASH.update("", phase="COMPLETE")
        _dl("Resume complete")

    _print_final(final)


def _do_resprint(plan_path: "Path"):
    """Re-run all sprints from a saved Demis strategy plan, ignoring previous results."""
    global _DASH, _CP

    plan_text = plan_path.read_text(encoding="utf-8", errors="replace")
    task_map = {
        "Radi":     _parse_tasks(plan_text, "Radi"),
        "Casandra": _parse_tasks(plan_text, "Casandra"),
        "Viktor":   _parse_tasks(plan_text, "Viktor"),
    }

    _CP = Checkpoint()
    _CP.data["plan"]     = plan_text
    _CP.data["task_map"] = {k: v for k, v in task_map.items()}
    _CP.mark("council")
    _CP.mark("synthesis")
    _CP.mark("plan_review")

    _DASH = Dashboard()
    _DASH.register(ALL_AGENTS)
    with _DASH:
        approved_map = task_map  # all leads approved
        alpha_sum, beta_sum, cto_sum, risk_sum = _run_sprints(plan_text, approved_map)
        final = demis_final_report(plan_text, alpha_sum, beta_sum, cto_sum, risk_sum)
        _DASH.update("", phase="COMPLETE")
        _dl("Re-sprint complete")
    _print_final(final)


def _run_from_last_result(session: dict):
    """
    Resume an incomplete session from scanned project_output artifacts.
    Preloads checkpoint with all already-completed member/sprint work so
    run_team_sprint skips them automatically.
    """
    global _DASH, _CP

    result_path = session["result_path"]
    plan_path   = session.get("plan_path")
    transcript  = result_path.read_text(encoding="utf-8", errors="replace")

    # ── Load or re-synthesize plan ─────────────────────────────────
    if plan_path:
        plan_text = plan_path.read_text(encoding="utf-8", errors="replace")
        task_map  = {
            "Radi":     _parse_tasks(plan_text, "Radi"),
            "Casandra": _parse_tasks(plan_text, "Casandra"),
            "Viktor":   _parse_tasks(plan_text, "Viktor"),
        }
        print(f"\n  [Plan loaded from {plan_path.name}]")
        for lead, tasks in task_map.items():
            print(f"    {lead}: {len(tasks)} tasks")
    else:
        print("\n  [No saved plan found — asking Demis to re-synthesize...]")
        _CP = Checkpoint()
        _CP.data["transcript"] = transcript
        _CP.save()
        _DASH = Dashboard()
        _DASH.register(ALL_AGENTS)
        with _DASH:
            plan_text, task_map = demis_synthesize(transcript)
        _DASH = None

    # ── Build checkpoint preloaded with all done work ──────────────
    _CP = Checkpoint()
    _CP.data.update({
        "transcript":    transcript,
        "plan":          plan_text,
        "task_map":      task_map,
        "phases_done":   list(session.get("phases_done", ["council", "synthesis"])),
        "member_results": session.get("member_results", {}),
        "summaries":     session.get("summaries", {"alpha": "", "beta": "", "cto": ""}),
    })
    _CP.save()

    done_m = list(session.get("member_results", {}).keys())
    if done_m:
        print(f"\n  [Checkpoint preloaded: {len(done_m)} members already done → will be skipped]")
        for key in done_m:
            print(f"    ✓ {key.split(':')[1]}")

    # ── Recover already-done sprint summaries ──────────────────────
    sums     = session.get("summaries", {})
    alpha_sum = sums.get("alpha", "") or "[SKIPPED]"
    beta_sum  = sums.get("beta",  "") or "[SKIPPED]"
    cto_sum   = sums.get("cto",   "") or "[SKIPPED]"

    _DASH = Dashboard()
    _DASH.register(ALL_AGENTS)

    with _DASH:
        # Plan review always shown so user can adjust before pending sprints run
        approved_map = review_plan(plan_text, task_map)

        if approved_map is not None:
            phases = _CP.data["phases_done"]

            lr_results: dict[str, str] = {}
            lr_lock = threading.Lock()

            def _lrsprint(key, fn, *args):
                r = fn(*args)
                with lr_lock:
                    lr_results[key] = r

            lrthreads = []
            if "alpha_sprint" not in phases and approved_map.get("Radi") is not None:
                lrthreads.append(threading.Thread(target=_lrsprint, daemon=True,
                    args=("alpha", run_team_sprint, RADI, ALPHA_MEMBERS, plan_text, approved_map.get("Radi", []))))
            elif "alpha_sprint" in phases:
                _dl("Team Alpha sprint already complete (loaded from output)")

            if "beta_sprint" not in phases and approved_map.get("Casandra") is not None:
                lrthreads.append(threading.Thread(target=_lrsprint, daemon=True,
                    args=("beta", run_team_sprint, CASANDRA, BETA_MEMBERS, plan_text, approved_map.get("Casandra", []))))
            elif "beta_sprint" in phases:
                _dl("Team Beta sprint already complete (loaded from output)")

            if "cto_sprint" not in phases and approved_map.get("Viktor") is not None:
                lrthreads.append(threading.Thread(target=_lrsprint, daemon=True,
                    args=("cto", run_viktor_solo, plan_text, approved_map.get("Viktor", []))))
            elif "cto_sprint" in phases:
                _dl("Viktor CTO sprint already complete (loaded from output)")

            for t in lrthreads: t.start()
            for t in lrthreads: t.join()

            alpha_sum = lr_results.get("alpha", alpha_sum)
            beta_sum  = lr_results.get("beta",  beta_sum)
            cto_sum   = lr_results.get("cto",   cto_sum)

        # Risk debate if sprints ran
        if any(s not in ("[SKIPPED]", "") for s in [alpha_sum, beta_sum, cto_sum]):
            risk_sum = run_risk_debate(plan_text, alpha_sum, beta_sum, cto_sum)
        else:
            risk_sum = "[SKIPPED]"
        final = demis_final_report(plan_text, alpha_sum, beta_sum, cto_sum, risk_sum)
        _DASH.update("", phase="COMPLETE")
        _dl("Resume complete")

    _print_final(final)


if __name__ == "__main__":
    main()
