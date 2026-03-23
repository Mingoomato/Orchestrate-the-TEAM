# Orchestrate the Team

A multi-agent AI orchestration system for quantitative trading research. A team of specialized AI agents — each with a distinct role — collaborates to analyze strategies, run backtests, validate results, and produce actionable reports.

---

## Why This Exists

Building a quant trading system involves too many parallel workstreams for a single agent to handle well: strategy design, implementation, statistical validation, risk management, and infrastructure. This orchestrator runs a structured **Grand Council + Sprint** workflow where each agent owns a specific domain, executes real code via the Claude CLI, and reports actual results — not hallucinated ones.

---

## Architecture

### Agents

| Codex | Name | Role |
|-------|------|------|
| CEO | Demis | Strategy synthesis, final report |
| C0 | Viktor | CTO — mathematical validation, OOS methodology, Gate audit |
| C1 | Radi | Team Alpha Lead — alpha signal design, backtest scan |
| C2 | Casandra | Team Beta Lead — systems architecture, infrastructure |
| C3 | Darvin | ML engineer — QLSTM training, model architecture |
| C4 | Felix | Data engineer — Bybit API, OHLCV pipeline |
| C5 | Jose | Risk manager — kill switch, position sizing |
| C6 | Felipe | Backtest engineer — WR/MDD/Sharpe, OOS analysis |
| C7 | Marvin | Quantum computing — VQC circuits, quantum layers |
| C8 | Schwertz | Strategy analyst — signal logic, entry/exit rules |
| C9 | Finman | Performance engineer — CUDA optimization |
| C10 | Ilya | Integration synthesizer — cross-team coordination |

### Execution Model

- **Part A (Analysis)** — Gemini 2.5 Pro via API: strategy discussion, synthesis, planning
- **Part B (Execution)** — Claude Code CLI subprocess: actual bash commands, file reads/writes, backtests

This split ensures execution tasks produce real file outputs, not simulated ones.

---

## Workflow Phases

```
agenda.md
    │
    ▼
Phase 0.5 — Pre-Council Results Scan
    Radi: reads latest backtest CSVs → WR, EV, MDD summary
    Viktor: audits Gate 1/2 status, checkpoint state
    │
    ▼
Phase 1 — Grand Council (data-driven, no debate)
    Viktor + Radi + Casandra: opening position statements
    Demis: synthesizes into strategy plan
    │
    ▼
Phase 2.5 — Plan Review
    User approves task assignments before sprints fire
    │
    ▼
Phase 3/4 — Team Sprints (parallel)
    Alpha Team (Radi leads): alpha research, backtest execution
    Beta Team (Casandra leads): infrastructure, data pipeline
    Viktor Solo: mathematical validation, Gate 1/2 testing
    │
    ▼
Phase 4.7 — Viktor Risk Audit
    Reads actual r-multiple CSVs
    Computes Kelly fraction, Gate 1/2 verdict
    Outputs: INCREASE / MAINTAIN / REDUCE
    │
    ▼
Phase 5 — Demis Final Report
    Reads all sprint outputs, writes report to project_output/
```

---

## Setup

```bash
# 1. Install dependencies
pip install google-generativeai

# 2. Set Gemini API key
export GEMINI_API_KEY=your_key_here
# or
export GOOGLE_API_KEY=your_key_here

# 3. Ensure Claude CLI is installed and authenticated
claude --version

# 4. Write your agenda
echo "# Grand Council Agenda\nYour agenda here..." > agenda.md

# 5. Run
python orchestrate.py
```

---

## Session Options

When you run `python orchestrate.py`, you are prompted with the current session state:

```
r / resume    → resume from last checkpoint (skip completed phases)
s / re-sprint → re-run all sprints from saved strategy plan
n / new       → set new agenda, run fresh Grand Council
```

Use `s` to re-run only the sprint phase when the council plan is already good but execution needs a retry.

---

## Output Files

All outputs are written to `project_output/`:

| File pattern | Producer | Contents |
|---|---|---|
| `demis_strategy_plan-*.md` | Demis | Sprint task assignments per team |
| `*_initial-*.md` | Each member | Implementation result |
| `cto_validation_report-*.md` | Viktor | Gate 1/2 test results |
| `risk_audit_report-*.md` | Viktor | Kelly fraction + position verdict |
| `final_report-*.md` | Demis | Full session summary |

Backtest CSVs (r-multiples, equity curves) are written to `reports/`.

---

## Validation Gates

Gate 1 and Gate 2 are defined in `backtesting/validation.py`:

- **Gate 1**: Win rate > 25.4% (fee-adjusted breakeven for TP=3×ATR, SL=1×ATR, 0.375% round-trip cost)
- **Gate 2**: Bootstrap p-value < 0.05 on mean R-multiple > 0, minimum 250 trades

No strategy proceeds to live consideration without passing both gates.

---

## Cross-Session Memory

Session results are persisted to SQLite and optionally synced to GitHub on exit (`Ctrl+C`). On startup, the last session is detected and loaded for resume/re-sprint options.
