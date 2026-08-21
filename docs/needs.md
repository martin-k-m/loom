# What loom needs from twill

loom is written in twill and it runs: `twill test tests` passes eight suites and
`twill run examples/classifier.tw` trains a model. This file is no longer the
reason it does not run. It is the record of what this library asked the language
for, with the file and function that needed each one, what loom did in the
meantime, and, for the ones that have since arrived, whether loom has taken them
up. An entry the language delivered and loom has not wired up says exactly
that, because "twill cannot" and "loom has not" are different sentences and only
one of them is a language work item.

It is a work queue for the language, not a complaint. Every entry was reached by
writing real code and hitting the wall, which is the only way a list like this
is worth anything. Where loom has a workaround the workaround is described,
because how ugly a workaround is measures how badly the feature is wanted.

The baseline is milestone 1 of `docs/self-hosting.md` in the twill repository:
`mode systems`, `I64` with bitwise operations, `Str` with length and byte
indexing, `Arr[T]`, `Dict[Str, V]`, `struct`, and `read_file`. Everything below
is on top of that.

## Was blocking: loom could not run at all without these

### 1. `mode systems` itself

**Used by:** every file
**Status:** DELIVERED in twill 1.6. Closed.

Nothing else on this list mattered until this did. It landed, and everything
under `src/` and `tests/` has run against a released twill since.

### 2. A type for a parameter tree

**Needs:** a spelling for "a tensor, or a list or record nesting tensors"
**Used by:** `src/state.tw` (`OptState`, `Run`, `StepResult`), `src/trainer.tw`
(`fit`, `evaluate`, `predict`, `apply`), `src/checkpoint.tw` (`Checkpoint`)
**Status:** DELIVERED. `Tree` is the name. loom writes it in every signature
that takes a parameter tree and the code checks and runs, so the guess below was
right and the rest of this entry is kept as the record of what was asked for.

loom writes `Tree` everywhere a parameter tree appears and assumed it would be
the name. It is the single most-used type in this repository, it is the type
`std/optim` is already generic over, and systems mode makes annotations
mandatory, so every one of those functions is currently unwritable as specified.

If `Tree` is not the answer, the alternatives loom can work with are a builtin
generic (`Tree[Tensor]`), or dropping the annotation requirement for parameters
whose type is a tree. What loom cannot work with is naming a concrete structure,
because the whole point of `map_leaves` is that the framework does not know the
model's shape.

### 3. Function values as parameters and struct fields

**Needs:** a function type in an annotation, and a function value passed to and
stored by a systems-mode function
**Used by:** `src/trainer.tw` (`fit` takes `step` and `eval_batch`; `predict`
takes `forward`; `default_step` takes `loss_fn`)
**Status:** DELIVERED. A systems-mode function takes a function value and the
type is spelled `fn(A, B) -> C`. `tr.fit` takes `step` and `eval_batch` that
way and `examples/classifier.tw` trains through both.

loom writes `step: fn(Tree, st.OptState, Tensor, Tensor, F64) -> st.StepResult`
and that is the syntax. This is not a convenience: the explicit step function
is the design. A trainer that hid the update rule inside itself would be the
thing loom exists not to be, and without function parameters there is no way to
hand one in.

The related want is a function value in a struct field, which is what would
turn `src/callback.tw` from a tagged struct into a record of closures. See
entry 5.

### 4. Multiple return values, or `Res[T, E]`

**Used by:** `src/trainer.tw` (`fit`, `resume`), `src/checkpoint.tw`
(`restore`), `src/callback.tw` (`validate`, `unpack_state`)
**Status:** done for `Res[T, E]` (2026-08, on twill 1.7); multiple return values
are still not designed anywhere and are a separate want.

The five fallible functions returned a `Str` that was empty on success. They
return `Res[Unit, Str]` now, and they were moved together, which is what this
entry said had to happen: they call each other, and two conventions in one
repository is worse than either.

`fit` shows what it bought. It opened with

    let bad = cb.validate(cbs_in)
    if len(bad) > 0 { return bad }

which is a propagation written out by hand, and is now `cb.validate(cbs_in)?`.
`resume` had the same shape around `restore` and lost it the same way.

**The multiple-returns half of this entry stands, and `Res` is not the answer to
it.** `take` returns a `Batch` and `split` returns a `Split`, and this entry
called both "a tuple with a name". They are -- but neither function can fail, so
there is no failure channel to remove and wrapping them in a `Res` would say
something untrue. What they want is a way to return two values, which twill
still does not have. `Batch` and `Split` therefore stay exactly as they were,
and the reason is recorded here so the next reader does not convert them.

### 5. Sum types and `match`

**Used by:** `src/callback.tw` (`Callback.kind`, `fire`), `src/state.tw`
(`OptState.kind`), `src/callback.tw` (`Callback.sched`)
**Status:** landed in twill 1.6, and loom uses it. Closed.

Four discriminants in this repository were I64 constants: callback kind,
schedule shape, optimiser kind, and precision policy. All four are enums now,
and every dispatch over them is an exhaustive `match`. What that bought, in each
case, is that the if-chain's fallthrough is gone: `cb.fire` fell off the end for
an unhandled kind, `tr.apply` fell through to adam, and `pre.narrow` fell
through to no narrowing at all. Each of those is a run that trains and is not
the run that was asked for.

Two things are unchanged and are not waiting on anything. `Callback` is still
one flat struct holding the union of every callback's fields, because a case
carries one payload and the checkpoint packs callback state positionally, which
wants a fixed field set. So the correctness cost named here still stands: a
checkpoint callback carries `patience` and `min_delta` fields it never reads,
and nothing stops a caller setting them and expecting an effect. That is entry 3
(function values in struct fields), not this one.

The two discriminants that reach a file on disk, `OptState.kind` and
`Callback.kind`, are written as integers and converted at the boundary by
`st.opt_code`/`st.opt_of_code` and `cb.kind_code`. The numbering is unchanged, so
checkpoints written before this survive it.

### 6. Reading and restoring the generator position

**Needs:** `rng_state() -> I64` and `set_rng_state(I64)`, or a generator value
that can be held rather than a global one
**Used by:** `src/rng.tw`, and therefore `src/trainer.tw` and
`src/checkpoint.tw`
**Status:** DELIVERED in twill 1.7, and loom has not taken it up. The language
grew the better of the two answers this entry asked for: a held generator,
`rng_open(seed)` with `rng_close`, `rng_f64`, `rng_u53`, `rng_uniform`,
`rng_norm`, `rng_normal` and `rng_perm` drawing from it. `seed(n)` and the
global stream are still there and are still what `src/rng.tw` calls.

The rest of this entry describes the workaround loom is still running, and it
is now a workaround for a gap that is loom's rather than twill's.

A resumed run must draw the same numbers the uninterrupted run would have, and
the generator's position after ten epochs is not recoverable from the seed alone
without replaying them. loom works around this by reseeding at the top of every
epoch from `mix(base, epoch)`, which makes epoch 10 identical whether it is the
tenth epoch of a run or the first after a resume.

The workaround is good enough to be defensible and it has a real cost, stated in
`src/rng.tw`: a run can only be checkpointed on an epoch boundary. Mid-epoch
checkpointing needs this entry.

A held generator would be better than a readable global one. A global generator
means an evaluation pass that drew a single random number would shift every
subsequent training batch, and the only reason `src/trainer.tw` gets away with
not worrying about that is that its evaluation path draws nothing. That is a
property maintained by inspection, which is the wrong way to maintain anything.

## Was blocking: features the source assumed exist

### 7. I64 bitwise operators, spelled

**Needs:** `xor`, `and`, `shr` on I64, and their spelling fixed
**Used by:** `src/rng.tw` (`mix`, `epoch_seed`, `derive`)
**Status:** section 1.2 promises "bitwise `and or xor shl shr not` on I64" and
does not say whether they are infix operators, builtin functions, or methods.

loom writes them as builtin calls, `xor(a, b)` and `shr(z, 30)`, because `and`
and `or` are already the short-circuiting logical operators in numeric mode and
overloading them on I64 seems worse than a function.

`shr` has since been specified, in twill's `docs/language-guide.md` operators
section: it is an **arithmetic** shift, and twill's `docs/needs.md` NEEDS-85 is
the entry that settles it. loom used to assume the opposite, and that assumption
was wrong rather than merely unstated: splitmix64's finaliser is a different
generator under an arithmetic shift, so every seed derived from a value with its
top bit set was silently off the reference stream. `src/rng.tw` now carries a
`ushr` built the same way as twill's `src/float.tw` `ushr`, and
`tests/rng_test.tw` pins the finaliser and the seed stream to values taken from
a reference splitmix64. What is still wanted from the language is a native
logical shift so the helper can be deleted, not a change of semantics.

### 8. twill's terminal layer, reachable from a package

**Needs:** `src/term/` and `src/cli/` reachable from a package
**Used by:** `src/report.tw`, which now calls them
**Status:** DELIVERED for colour and capability detection, and taken up. The
stateful bar is still deliberately not adopted; see below.

The resolution was the one predicted here after all. The terminal layer is under
`std/` now: `std/term/caps`, `std/term/ansi`, `std/term/theme`, alongside
`color`, `width`, `box` and `frame`. They are ordinary `std/` modules and
therefore reachable from an installed package with no vendoring and no change to
the import rule. `src/report.tw` imports the first three, detects capabilities
and lights the loss value and the bar's fill, dropping to plain text the moment
the output is piped.

There is no `std/cli` and therefore no determinate bar with a rate and an ETA to
adopt. loom keeps its own stateless per-epoch bar, lit from the same palette so
the two never drift in colour. Building the estimate is entry 16, which is now a
loom work item rather than a language one.

### 9. `f64` and `i64` conversions, and `F64` as a declared type

**Used by:** `src/metrics.tw` (`update`, `count`), `src/report.tw` (`fixed`),
`src/callback.tw` (`lr_at`, `pack_state`)
**Status:** DELIVERED. `F64` is a declared type, `i64(f)` and `f64(n)` convert,
and `Meter.total` is an `F64` and not a rank-0 tensor, so the allocation
question raised below does not arise.

Systems mode makes annotations mandatory, so a metric value has to be spelled
something. loom writes `F64` and reads it as "a rank-0 tensor with the tensor
machinery off". If the answer is that a scalar in systems mode is still a
`Tensor`, then `Meter.total` is a tensor, every accumulation allocates, and an
epoch of accumulation builds a chain of them; that is a performance answer loom
would want to know about before the interpreter finds out.

### 10. Struct field mutation through a parameter

**Used by:** everywhere. `src/callback.tw` (`observe` writes `c.wait`),
`src/trainer.tw` (`fit` writes `r.st.epoch`), `src/metrics.tw` (`update`)
**Status:** DELIVERED, and it is the reference answer. `fit` advances the run it
was given, callbacks record into `State`, and `met.update` mutates a meter in
place; all three are covered by tests that pass.

loom depends on this more heavily than spool does. If a struct passed to a
function is copied, then `fit` cannot advance the run it was given, no callback
can record anything, and every function in `src/metrics.tw` has to return a new
meter. The whole shape of this codebase assumes the reference answer.

## Not blocking, but the source is worse without them

### 11. A generic sort or a comparison-function parameter

**Would improve:** `src/callback.tw` (`ordered`)
**Status:** no generic sort; see entry 3 for function parameters.

`ordered` is an insertion sort by `(order, index)` written out by hand. It is
correct and it is eleven lines that a `sort_by` would make one. spool has four
copies of the same problem, which makes five in the ecosystem.

### 12. `Dict` values that are not scalars

**Would improve:** `src/metrics.tw` (`MeterSet`)
**Status:** a `Dict[Str, V]` does hold a struct in twill 1.6, and `MeterSet` is
one `Dict[Str, Meter]` now. Closed.

`MeterSet` was two parallel `Dict[Str, F64]` plus an `Arr[Str]` of names, where
it wanted to be one `Dict[Str, Meter]`. The parallel dicts could go out of step
and only a convention kept them together. As one dict, `update_named` and
`value_named` are the existing `update` and `value` rather than a second copy of
the weighting rule this file exists to get right.

The `Arr[Str]` stays regardless of this entry. Report column order has to be
stable across runs and across machines, and depending on the dict's iteration
order for it would be depending on a property this source should not have to
know.

### 13. A tensor-shaped `Dict` key, or an `Arr[Tensor]`

**Would improve:** `src/trainer.tw` (`predict`)
**Status:** DELIVERED. `T` may be `Tensor`: `tr.predict` accumulates an
`Arr[Tensor]` and concatenates once, and `predict_returns_one_row_per_input_row_in_input_order`
in `tests/trainer_test.tw` runs it. The quadratic fallback below was not needed.

`predict` accumulates per-batch outputs in an `Arr[Tensor]` and concatenates
once. If `Arr` cannot hold tensors, the fallback is to concatenate as it goes,
which is quadratic in the number of batches and allocates the whole output once
per batch.

### 14. `continue`, and early exit from a nested loop

**Would improve:** `src/callback.tw` (`fire_*`), `src/metrics.tw` (`has`)
**Status:** `return` exists; `continue` is not in the language guide.

Every `fire_*` function opens with a guard that returns when the hook is not one
it handles. That reads acceptably. The dispatch loop in `src/trainer.tw` and the
linear scan in `met.has` both want `continue`, and write an `if` around the
whole body instead.

### 15. A test runner

**Would improve:** `tests/`
**Status:** DELIVERED. `twill test tests` collects `*_test.tw`, runs each in a
fresh interpreter and reports once, which is exactly what this entry asked for.
CI calls it and so does the README.

`tests/harness.tw` is not gone. `twill test` reports a file as passed or failed
and the harness is what names the individual assertions inside one, so the two
are complementary rather than one replacing the other. The duplication this
entry complained about, the same harness in loom and in spool, is unchanged, and
what would close it is a `std/test` rather than the runner.

### 16. Timing, for the progress estimate

**Needs:** a monotonic clock, `mono_ms() -> I64`
**Used by:** `src/report.tw`, which reports a fill with no estimate
**Status:** DELIVERED in twill 1.7, and loom has not taken it up. `mono_ns()`
returns a monotonic nanosecond count and `clock_now_ms()` a wall-clock
millisecond one. Both were checked against the 1.7.1 binary.

So this stops being a language work item and becomes a loom one, and the reason
it is still open is not the arithmetic. The estimate needs a start time held
across the run, and a run resumed from a checkpoint at epoch 10 has no start
time for the first ten epochs; extrapolating as if it did produces a confident
wrong number, which is worse than the blank this prints today. Threading a time
source through the trainer with that case handled is the change. Until it is
written, the README says loom has no estimate and says why, and it no longer
says twill has no clock, because twill does.

The same requirement is bobbin's first entry, and it should be satisfied once.

### 17. A dtype as a value, and a name for its type

**Needs:** a dtype storable in a struct field and passable as an argument, and
a type name for it in an annotation
**Used by:** `src/precision.tw` (`Policy`, `narrow`)
**Status:** twill's NEEDS-110 makes the seven dtype names contextual
identifiers, read as dtypes only where a constructor or `.to` expects one.
`docs/dtypes.md` there calls a dtype "a name in the term language", but no
annotation spelling exists and no other position reads one.

A precision policy is one value answering one question, what dtype the forward
pass runs in, so `Policy` wants a `compute: Dtype` field and `narrow` wants
`t.to(pol.compute)`. Neither is writable. The argument of `.to` is a position
where a bare name is read as a dtype, not an expression evaluated to one, and
even if it were, systems mode makes annotations mandatory and the type of a
dtype value has no name. loom works around it the way `OptState.kind` does: a
discriminant, `Prec`, and one arm of `narrow` per case with each dtype name
written out literally. Since twill 1.6 the discriminant is a real enum and the
arms are an exhaustive `match`, so a case added to `Prec` and not handled here
is a check-time error rather than a tensor that silently stays wide. That fixes
loom's half of the failure mode and not twill's: a new dtype *in twill* still
means finding every such chain by hand, because nothing connects the set of
dtypes to the set of cases.

Half of this landed after it was written: twill grew a `dtype(t)` builtin that
returns a tensor's dtype as a name string, so the read direction exists and a
policy can now assert what a tensor actually is, which is what `grads_finite`
and a mixed-precision sanity check want. The half that is still missing is the
one this entry is really about: a dtype as a value to store in `Policy.compute`
and hand to `.to`. `dtype(t)` gives a string, and the `.to` argument is a
contextual name and not a string, so the two do not yet meet. The if-chain
stays until they do.

The adjacent gap: nothing reads a dtype back off a tensor, so
`tests/precision_test.tw` asserts that narrowing happened by observing that
values rounded, a behavioural proxy for a property the runtime already holds.
twill's NEEDS-114 covers printing it; introspecting it is not written down
anywhere.

### 18. A value-level scaled backward

**Needs:** `backward_scaled` and `grads_finite` reachable from the level a
framework works at, or the tape exposed
**Used by:** `src/precision.tw` (`mixed_step`)
**Status:** twill's NEEDS-112 names `backward_scaled(tp, root, seed, scale)`
and `grads_finite(gs)`. The first takes a tape and a node, and loom never
holds either: `value_and_grad` is the entire autodiff surface a training
framework sees.

`mixed_step` reproduces `backward_scaled`'s arithmetic by multiplying the loss
by the scale inside the differentiated closure, which by the chain rule scales
every gradient by exactly the same factor, so the result is identical. The
reason this entry exists anyway is the reason NEEDS-112 gives for naming the
functions at all: a hand-rolled version that forgets the skip looks like it
works. loom's copy has the skip; the point of putting the loop in the runtime
is that the next caller's does too. Wanted: either a
`value_and_grad_scaled(f, scale)` beside `value_and_grad`, or the tape as a
value so `backward_scaled` is callable as specified.

There is a second wall behind the first: `backward_scaled` returns
`Res[Arr[Tensor], Str]`, and the subset has no `Res` (entry 4), so even with a
tape in hand the result is not consumable today.

### 19. The leaves of a tree, as an array

**Needs:** `leaves(tree) -> Arr[Tensor]`, or a fold over leaves
**Used by:** `src/precision.tw` (`leaves_of`, feeding `grads_finite`)
**Status:** `map_leaves` and `zip_leaves` exist; nothing enumerates.

`grads_finite` takes `Arr[Tensor]` and a gradient arrives as a tree.
`leaves_of` collects the leaves by abusing `map_leaves` for its visit order: a
helper pushes each leaf into a captured array and returns it unchanged, and
the mapped tree is thrown away. It works because arrays are captured by
reference (entry 10) and it reads like what it is. The shape of the problem is
entry 11's sort and entry 14's `continue` again: the subset has the traversal
and not the escape hatch.
