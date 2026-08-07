# Changelog

## v0.1.0 (unreleased)

First cut of loom, the training framework for twill, written in twill.

It does not run. twill's `mode systems` is still being built. See
`docs/needs.md` for what is missing and `README.md` for the status table.
Nothing below has ever executed.

Added:

- `fit`, `evaluate` and `predict`, with the step function passed in by the
  caller so the update rule stays visible and replaceable.
- Seven hook points and a total callback order, with the two ordering rules
  checked by `cb.validate` at the start of a run rather than left to review.
  Two schedules, or two early stoppers, are refused with a message.
- Early stopping, periodic and best-only checkpointing, four learning-rate
  schedules (step, exponential, cosine with warmup, plateau), metric logging in
  human and JSON form, and a plain-text progress callback.
- Metric accumulation weighted by batch row count, so a short final batch does
  not distort an epoch mean, plus a ratio counter for metrics that are not
  means over rows.
- Checkpoint and restore covering parameters, optimiser moments, adam's step
  count, the epoch, the global step, the learning rate and the callbacks'
  patience counters. A restore into a different seed, batch size, optimiser
  kind or row count is refused.
- One explicit seed, threaded through a per-epoch derivation, so a resumed run
  reproduces an uninterrupted one exactly at epoch granularity.

Known gaps, deliberate for v0.1:

- No coloured output and no progress time estimate. twill's terminal layer is
  not reachable from an installed package; `docs/needs.md` entry 8.
- No mid-epoch checkpointing. The generator's position is not readable;
  `docs/needs.md` entry 6.
- No distributed training, no mixed precision, no gradient accumulation.
