# UPSTREAMS.md

Records the per-repo remotes and last-synced commit hashes from before the
nested git repos were consolidated into this single parent repo (2026-04-20).
Use these to re-clone a historical snapshot or pull official upstream changes.

---

## HunyuanVideo

- **path (pre-consolidation):** `HunyuanVideo/`
- **your fork:** `git@github.com:oagrawal/HunyuanVideo.git`
- **last pushed commit:** `da4cf9f` — "chore: wip snapshot before consolidation into parent repo" (2026-04-20)
- **note:** All work up to 2026-04-20 is preserved on the fork. DiCache results,
  MagCache calibration, and EasyCache VBench results were pushed in the final WIP snapshot.

---

## CogVideo

- **path (pre-consolidation):** `CogVideo/`
- **your fork:** `git@github.com:oagrawal/CogVideo.git`
- **last pushed commit:** `e8ef636` — "chore: wip snapshot before consolidation into parent repo" (2026-04-20)
- **note:** 1232 files committed (VBench scores, DiCache and EasyCache sweep results).

---

## Wan2.1 (outer wrapper repo)

- **path (pre-consolidation):** `wan/`
- **your fork:** `git@github.com:oagrawal/Wan2.1.git`
- **last pushed commit:** `6d58fe1` — "testing batching with 2 identical items in a batch" (2025-07-24)
- **note:** The outer `wan/` tracked early batching experiments. Its only
  uncommitted content was `Wan2.1/` itself (a nested repo), so nothing extra
  needed saving here.

---

## Wan2.1 (inner repo — the actual Wan2.1 codebase)

- **path (pre-consolidation):** `wan/Wan2.1/`
- **your fork:** `git@github.com:oagrawal/Wan2.1.git`
- **last pushed commit:** `bf32107` — "fresh start to wan2.1 + dicache" (2026-04-01)
- **uncommitted work NOT pushed:** `dicache_exp/run_wan_dicache.py` (modified),
  `dicache_results/` (untracked), `vbench_eval_easycache/` plots + scripts
  (untracked). Could not push — `.git/objects` subdirs owned by another user
  (`saarth`) blocked writing new objects. All content is preserved in the
  consolidated parent repo's working tree.
- **⚠️ revoke PAT:** The old remote URL embedded a GitHub PAT (now redacted).
  If not already revoked, do so at https://github.com/settings/tokens.

---

## mochi

- **path (pre-consolidation):** `mochi/`
- **your fork:** `git@github.com:oagrawal/mochi.git`
- **last pushed commit:** `8bf7ae6` — "chore: wip snapshot before consolidation into parent repo" (2026-04-20)
- **note:** TeaCache experiments, VBench eval scripts, and INSTRUCTIONS committed.

---

## HunyuanVideo/VBench (official VBench repo — not a fork)

- **path (pre-consolidation):** `HunyuanVideo/VBench/`
- **upstream:** `https://github.com/Vchitect/VBench.git`
- **pinned commit:** `07bc8a4` (branch `master`)
- **note:** Pure upstream clone, no local modifications. To update:
  `cd HunyuanVideo/VBench && git fetch origin && git merge origin/master`
  (must first add safe.directory: `git config --global --add safe.directory <path>`)
