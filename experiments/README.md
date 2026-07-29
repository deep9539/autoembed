# Experiments (paper ablations & variants)

Canonical benchmark tasks live in `configs/` — frozen, released, comparable.
This folder holds **paper-only variants** that change a *task* dimension:
split fractions, task-set tweaks, or one-off ablation specs.

Base-model sweeps do NOT belong here. Vary the base at runtime with
`--base` / `AUTOEMBED_BASE_MODEL` and pin it with `--base-revision` /
`AUTOEMBED_BASE_REVISION`; both are recorded in each run’s `meta.json`. Reserve
this folder for genuine task changes, not base, agent, or budget changes.

`reasoning-bright.json` is a validated but non-canonical protocol fixture retained while
BRIGHT base/reference choices are deferred. It is not shown by `autoembed list`.
