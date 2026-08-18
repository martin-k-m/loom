<p align="center">
  <img alt="twill" src="https://raw.githubusercontent.com/twill-lang/twill/main/assets/twill-mark.png" width="120">
</p>

<h1 align="center">loom</h1>

<p align="center">
  <b>The training framework for <a href="https://github.com/twill-lang/twill">twill</a>.</b><br>
  Written in twill.
</p>

<p align="center">
  <img alt="loom" src="https://img.shields.io/badge/loom-v0.1.0-7FE3C4?style=flat-square&labelColor=12332C">
  <img alt="written in twill" src="https://img.shields.io/badge/written%20in-twill-D2F0E4?style=flat-square&labelColor=12332C">
  <img alt="status" src="https://img.shields.io/badge/tests-passing-4FB79B?style=flat-square&labelColor=12332C">
  <img alt="MIT" src="https://img.shields.io/badge/license-MIT-D2F0E4?style=flat-square&labelColor=12332C">
</p>

---

## It runs

`loom` is written in twill, in `.tw` files, using `mode systems`. That subset
did not exist when this library was written, so for a long time none of the code
here executed and this section said so. twill 1.6 is the release that closed it:
the 8 test suites under `tests/` pass, and CI runs them against a released
twill on every push rather than gating on the prose in this file.

```bash
twill test tests
```

You need twill 1.6.0-rc2 or newer. `docs/needs.md` is still worth reading -- it
is the list of what this library asked the language for, and it now records
which of those arrived and which are still open.

## What loom is

The layer above `std/nn`. `std/nn` gives you layers, activations and losses;
`std/optim` gives you an update rule. Neither gives you an epoch, a metric that
survives a short final batch, a checkpoint you can resume from, or a place to
hang early stopping. That is loom.

| Piece | State |
| --- | --- |
| `fit` / `evaluate` / `predict`, with the step function passed in | written, unrun |
| Callbacks with a total, documented order and seven hook points | written, unrun |
| Early stopping, checkpointing, LR schedules, metric logging, progress | written, unrun |
| Checkpoint and restore covering parameters, optimiser state and epoch | written, unrun |
| Batch-size-weighted metric accumulation | written, unrun |
| Reproducibility from one explicit seed, threaded, no hidden global | written, unrun |
| Mixed precision: bf16 and f16 policies, f32 masters, dynamic loss scaling | written, unrun |
| Coloured output and a progress bar with a time estimate | **not possible.** See below |
| Distributed training, gradient accumulation | **not in v0.1** |
| Anything running end to end | **no** |

## The loop is not hidden

loom does not own your update rule. `fit` takes a step function, and the step
function is the whole update: forward, loss, gradient, optimiser, new
parameters. loom owns the epoch, the batching, the metrics, the callbacks and
the checkpoint, and nothing else.

```rust
mode systems

import "twill_modules/loom/src/trainer.tw" as tr
import "twill_modules/loom/src/state.tw" as st
import "twill_modules/loom/src/data.tw" as dt
import "twill_modules/loom/src/metrics.tw" as met
import "twill_modules/loom/src/callback.tw" as cb

import "std/nn" as nn

fn logits(p: Tree, x: Tensor) -> Tensor {
  let h = relu(x @ transpose(p.w1) + p.b1)
  h @ transpose(p.w2) + p.b2
}

fn loss_fn(p: Tree, x: Tensor, y: Tensor) -> Tensor {
  tr.cross_entropy_batch(logits(p, x), y, 3)
}

# The step. Every gradient in the run passes through here, in your file.
fn step(p: Tree, opt: st.OptState, x: Tensor, y: Tensor, lr: F64) -> st.StepResult {
  tr.default_step(p, opt, x, y, lr, loss_fn)
}

# What a validation pass measures. Per-row means, so the trainer can weight
# them by the batch's row count.
fn eval_batch(p: Tree, x: Tensor, y: Tensor) -> met.MeterSet {
  let ms = met.new_meters()
  let out = logits(p, x)
  met.update_named(ms, "loss", item(tr.cross_entropy_batch(out, y, 3)), shape(x)[0])
  met.update_named(ms, "accuracy", met.accuracy(out, y), shape(x)[0])
  ms
}

let params = { w1: nn.he_init(16, 4), b1: zeros(16),
               w2: nn.he_init(3, 16),  b2: zeros(3) }

# epochs, batch size, learning rate, seed. The seed is not optional.
let cfg = st.config(60, 16, 0.01, 20260807)
let run = st.new_run(params, st.adam(params), cfg)

let cbs: Arr[cb.Callback] = [
  cb.cosine_lr(0.01, 0.0002, 60, 3),
  cb.early_stopping("accuracy", true, 8, 0.001),
  cb.checkpointing("runs/blobs.ckpt", 10, true, "accuracy", true),
  cb.metric_log(false),
  cb.progress(24, 0),
]

let err = tr.fit(run, cbs, train, val, step, eval_batch)
if len(err) > 0 { print("loom: " + err) }

let pred = tr.predict(run.params, val.x, 64, logits)
```

Output, one line per epoch:

```
epoch  1/60  loss 1.0421  lr 0.002500  val_loss 0.9980  val_accuracy 0.4167
epoch  2/60  loss 0.9013  lr 0.005000  val_loss 0.8442  val_accuracy 0.6667
...
epoch 41/60  loss 0.0942  lr 0.000412  val_loss 0.1180  val_accuracy 0.9583
stopped: accuracy has not improved for 9 evaluations; best was 0.9583 at epoch 32
```

`examples/classifier.tw` is the same program, complete.

## Callback ordering

Ambiguous callback ordering is a real source of silent bugs, so loom's is total
and it is written down. It is not the order you list them in.

**Hook points.** Seven, and no others.

| Hook | When |
| --- | --- |
| `HOOK_RUN_BEGIN` | once, before the first epoch, after any restore |
| `HOOK_EPOCH_BEGIN` | after the epoch seed is set and the row order drawn |
| `HOOK_BATCH_BEGIN` | before each optimiser step; `State.lr` set here is the rate the step uses |
| `HOOK_BATCH_END` | after each step; `train_loss` is the running epoch mean, not the batch loss |
| `HOOK_EVAL_END` | after a validation pass, only on epochs where one ran |
| `HOOK_EPOCH_END` | after the epoch and any evaluation; `State.epoch` already incremented |
| `HOOK_RUN_END` | once, however the run ended |

**Order.** Every callback carries a band. Within a hook, callbacks run by
ascending band, ties broken by position in your array. Begin hooks run in that
order; end hooks run in the reverse of it, so a callback that wraps another at
`BATCH_BEGIN` still wraps it at `BATCH_END`.

| Band | Callback | Why there |
| --- | --- | --- |
| 10 | schedules | writes `State.lr`, so everything after it sees the rate this epoch trains at |
| 20 | early stopping | writes `State.stop`, and owns "is this the best epoch" |
| 30 | checkpointing | asks early stopping that question, so it runs after it |
| 40 | metric logging | reads only; can name the file the checkpoint just wrote |
| 50 | progress | reads only, prints last |

**The rule that keeps it maintainable.** A callback that writes to `State` has a
lower band than every callback that reads what it wrote, and two callbacks never
write the same field. Both are checked by `cb.validate` at the start of the run,
not left to review: two schedules or two early stoppers in one run are refused
with a message rather than resolved by whichever happens to sort first.

**A consequence worth knowing.** A metric-monitoring callback lives on
`HOOK_EVAL_END`, not `HOOK_EPOCH_END`, because on `EPOCH_END` a validation
metric may be an epoch stale. It also means patience is counted in evaluations,
so with `eval_every = 2` a patience of 3 is six epochs.

## Metrics that survive a short final batch

The classic bug: collect one loss per batch, take their mean. That is an
unweighted mean of weighted means, and it is only correct when every batch is
the same size. With 1000 rows and a batch size of 256 the last batch holds 232
and gets the same say as each 256-row batch.

`src/metrics.tw` stores a weighted sum and the total weight, and the weight is
the batch's row count. That is the only thing that is correct.

It is also only correct for metrics that are themselves means over rows: loss,
accuracy, MAE. Precision, recall, F1 and AUC are not, and they get `Counter`,
which accumulates the counts and divides once at the end.

## Checkpoints that resume rather than restart

The test loom is written against: train 20 epochs; separately train 10,
checkpoint, restore, train 10 more. The two parameter trees must be equal.
`tests/trainer_test.tw` is that test.

**Captured:** parameters, optimiser moments, adam's step count `t`, the epoch,
the global step, the current learning rate, the base seed, and the callbacks'
patience counters and bests.

`t` is why a parameters-only checkpoint is wrong. Adam divides by `1 - b1^t`,
and restarting `t` at 1 multiplies the first resumed update by about ten. The
loss spike gets blamed on the data.

**Not captured:** the dataset (the row count is recorded and a mismatch is
refused), the model's shape, the generator's position within an epoch, and the
step function, which is a function value and not serialisable. A restore into a
different seed, batch size or optimiser kind is an error, not a warning.

Checkpoints are taken on epoch boundaries only. That is a consequence of how
reproducibility is done, below, and it is stated rather than worked around.

## One seed, threaded

Deterministic-by-default randomness makes a program reproduce. It does not make
a *resumed* run reproduce: the generator has a position as well as a seed, and a
run restarted at epoch 10 has a generator at position zero while the
uninterrupted run's is wherever ten epochs of shuffling left it. Every parameter
after that differs.

loom reseeds at the top of every epoch from a mix of the base seed and the epoch
index, so epoch 10 draws the same numbers whether it is the tenth epoch of a run
or the first after a resume. Nothing in loom reads a global seed, and nothing
outside `src/rng.tw` calls `seed`. Other stochastic decisions, a train/test split
or an initialisation, draw from a *derived* seed, so adding a validation split
does not shift the batch orders of a run that previously had none.

The cost: within an epoch the stream is still positional, which is why
checkpoints are on epoch boundaries.

## Mixed precision, and why bf16

twill's dtype design (`docs/dtypes.md` in the twill repository) gives training
three rules: the forward pass may run narrow, gradients are never narrower than
f32, and anything narrower than f32 accumulates in f32. `src/precision.tw`
turns those into a policy. Adopting one is one changed line in a step function
and one call on the initial parameters:

```rust
import "twill_modules/loom/src/precision.tw" as prec

let POLICY = prec.mixed_bf16()

fn step(p: Tree, opt: st.OptState, x: Tensor, y: Tensor, lr: F64) -> st.StepResult {
  prec.mixed_step(POLICY, p, opt, x, y, lr, loss_fn)
}

let params = prec.masters(init_model())
```

`masters` makes the parameters f32. `mixed_step` narrows them, and the batch,
to the policy's dtype for the forward pass, so the activations and the matmuls
that move the memory run narrow; twill's autodiff hands the gradients back at
f32, and the optimiser updates the f32 masters, which are what checkpoints
capture. The narrow weights are a rounded copy, remade every step. Evaluate
with the same policy: an `eval_batch` that runs the masters wide measures a
model the run is not training.

**Use `prec.mixed_bf16()`.** bf16 and f16 are the same sixteen bits spent
differently. bf16 keeps f32's exponent range and about three significant
digits, so every gradient representable in f32 is representable in bf16 and it
trains with no extra machinery. f16 carries about half a digit more precision
in a range that ends at 65504 and loses gradients below f16's smallest normal,
so it cannot train bare: a gradient that underflows to zero is a parameter
that stops learning and reports nothing.

`prec.mixed_f16()` therefore runs dynamic loss scaling: the loss is multiplied
by a scale before the backward pass, which by the chain rule multiplies every
gradient by the same factor and lifts it into representable range; the scale is
divided back out before the optimiser sees anything. When a gradient still
comes back non-finite the step is **skipped** and the scale halved; after 2000
clean steps the scale doubles, probing for the largest value the model
tolerates. The skip is the part that matters. Clipping an infinity to a large
finite number is a plausible-looking update in an arbitrary direction, and a
run built on it trains to a worse model while reporting nothing. A skipped
batch costs one batch, and `Policy.skipped` counts them.

So: f16's extra half digit rarely pays for the scaling machinery, and the
machinery is only as good as the run that remembers all of it. That is the
twill design's recommendation, and it is loom's.

Two limits, stated rather than discovered. Until twill's packed buffer lands
(NEEDS-111 there), a policy changes the arithmetic, exactly as a 16-bit run
would compute it, and saves no memory. And the loss-scale state is not
checkpointed: a resumed f16 run re-converges its scale rather than restoring
it, so it is not bit-identical to an uninterrupted run while the scales
differ. bf16 has no such window, which is one more reason it is the default.

## No colour, and no progress estimate

Stated plainly because it is a gap and not a decision. twill has a real terminal
layer in `src/term/` and a determinate progress bar in `src/cli/progress.tw`,
with capability detection, the palette and a smoothed time estimate. loom cannot
call them: twill resolves a non-`std/` import as a path relative to the
importing file, so only modules under `std/` are reachable from an installed
package, and the terminal layer is not one.

loom therefore prints plain uncoloured lines, one per epoch, and its `bar` is a
fill with no time remaining. The useful part of a progress bar for a 400-epoch
run is the estimate, so this is worse than what twill already has.

The fix is not to widen the import rule. It is to promote capability detection,
the palette and the bar into `std/term`, at which point loom deletes its
formatting helpers. `docs/needs.md` entry 8. Duplicating the bar here was the
obvious alternative and was rejected: two progress bars in one ecosystem drift,
and the drift is visible to users.

## Install

Once spool and `mode systems` both work:

```
spool add loom https://github.com/twill-lang/loom
```

spool vendors into `twill_modules/`, and twill's import is a path, so the import
lines are the long ones in the example above and they resolve relative to the
project root. That is twill's rule rather than loom's; see spool's README.

## Repository layout

```
src/state.tw        Config, State, Run, OptState, StepResult
src/data.tw         Dataset, batching, a seeded split
src/metrics.tw      weighted meters, meter sets, ratio counters
src/rng.tw          the seed, derived and threaded
src/precision.tw    the precision policy: masters, narrowing, the scaled step
src/callback.tw     hook points, the ordering, and the five callbacks
src/checkpoint.tw   what is captured, what is not, and the refusals
src/report.tw       fixed-width formatting, a human line and a JSON line
src/trainer.tw      fit, evaluate, predict, default_step, resume
tests/              tests, named as sentences
examples/           a complete three-class MLP
docs/needs.md       what the language still has to provide
```

## Dependencies

twill, and nothing else. No third-party twill packages and no Go. `std/nn`,
`std/optim`, `std/data` and the tensor builtins are the whole surface loom
builds on.

## Contributing

The most useful contribution right now is not code. It is a correction to
[`docs/needs.md`](docs/needs.md): a feature listed there that the language
already has, a workaround that is worse than described, or a missing entry found
by reading the source. After that, the callback ordering table above is the part
most worth arguing with.

## License

MIT. See [LICENSE](LICENSE).
