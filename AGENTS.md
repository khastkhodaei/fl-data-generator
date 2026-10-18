# AGENTS.md

This document defines the agent skills configuration and working agreements for automated development on `fl-data-generator`.

## Agent skills

### Issue tracker

Issues and specs are tracked via GitHub Issues using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles mapped to GitHub labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repository layout (`GLOSSARY.md` and `docs/adr/`). See `docs/agents/domain.md`.

## Working agreements

1. One task = one branch = one PR. Never merge your own PR.
2. Red-green-refactor: write the failing test first, paste its failure, then implement.
3. Touch only files listed in the task. No refactors, renames, or formatting changes outside scope.
4. Never change a public function signature or default value without asking first.
5. Before reporting, run `pytest -q --cov=fl_data_generator` and `ruff check .` and paste the raw output.
6. After two failed attempts at the same problem, STOP and report.
7. Every report ends with: files changed, tests before/after, coverage before/after, UNVERIFIED items, behavior changes that could break users.
