"""LingBot-Depth distillation auto-research harness.

Layout:
  prepare.py   IMMUTABLE — teacher, data, metrics, latency, objective.
  train.py     MUTABLE   — the student + distillation recipe (the agent edits this).
  run.py       runner    — train -> eval -> log -> keep/revert.
  program.md   the agent's steering document.
"""
