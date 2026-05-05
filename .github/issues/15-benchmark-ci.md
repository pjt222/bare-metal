---
title: "Automated benchmark CI with self-hosted runner or manual trigger"
labels: ["enhancement", "infrastructure", "ci"]
---

## Problem
Performance regressions are only detectable by running `scripts/bench_regress.py` locally on the GPU machine. No CI integration exists because:
1. GitHub Actions hosted runners lack GPUs
2. Self-hosted runners require always-on machine
3. Manual benchmark runs are easy to forget

## Proposed Solutions

### Option A: GitHub Actions + self-hosted runner on WSL machine
- Install GitHub Actions runner on the WSL Ubuntu machine
- Trigger on PRs that touch `.cu` files
- Runner executes `make test` + `scripts/bench_regress.py`
- Fails PR if regression >10%

**Pros**: Automatic, integrated into GitHub workflow  
**Cons**: WSL machine must stay online; security risk of self-hosted runners on personal machine

### Option B: Manual workflow dispatch (safer)
- GitHub Actions workflow with `workflow_dispatch` trigger only
- PR author or maintainer clicks "Run Benchmarks" button
- Runner executes only when explicitly requested

**Pros**: No always-on requirement, lower security surface  
**Cons**: Manual step, easy to skip

### Option C: Local-only with git hook
- Pre-push hook runs `scripts/bench_regress.py`
- Fails push if regression detected
- No GitHub dependency

**Pros**: Simple, no external infrastructure  
**Cons**: Only protects the pusher's knowledge, not PR reviewers

## Recommendation
Start with **Option C** (local hook), then evaluate **Option B** after 3 months of use.

## Files to Create
- `.githooks/pre-push` or `scripts/install-hooks.sh`

## Acceptance Criteria
- [ ] Local git hook installed and documented
- [ ] Hook runs `make test` + `bench_regress.py`
- [ ] Fails with clear message on regression
- [ ] Can be bypassed with `--no-verify` for WIP commits
