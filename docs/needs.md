# What loom needs from twill

loom is written in twill and does not run yet. This file is the reason: the
language and runtime features the source uses that twill does not provide
today, with the file and function that needs each one, and what loom does in
the meantime.

It is a work queue for the language, not a complaint. Every entry was reached by
writing real code and hitting the wall, which is the only way a list like this
is worth anything. Where loom has a workaround the workaround is described,
because how ugly a workaround is measures how badly the feature is wanted.

The baseline is milestone 1 of `docs/self-hosting.md` in the twill repository:
`mode systems`, `I64` with bitwise operations, `Str` with length and byte
indexing, `Arr[T]`, `Dict[Str, V]`, `struct`, and `read_file`. Everything below
is on top of that.

## Blocking: loom cannot run at all without these

### 1. `mode systems` itself

**Used by:** every file
**Status:** designed in `docs/self-hosting.md`, not implemented.

Nothing else on this list matters until this does.

### 2. A type for a parameter tree

**Needs:** a spelling for "a tensor, or a list or record nesting tensors"
**Used by:** `src/state.tw` (`OptState`, `Run`, `StepResult`), `src/trainer.tw`
(`fit`, `evaluate`, `predict`, `apply`), `src/checkpoint.tw` (`Checkpoint`)
**Status:** the concept exists at runtime (`map_leaves`, `zip_leaves`, and
`grad` following record structure) and has no name in the type language.

loom writes `Tree` everywhere a parameter tree appears and assumes it will be
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
**Status:** functions are values in numeric twill; whether a systems-mode
function may take one, and how the type is spelled, is not stated anywhere.

loom writes `step: fn(Tree, st.OptState, Tensor, Tensor, F64) -> st.StepResult`
and assumes that syntax. This is not a convenience: the explicit step function
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
**Status:** `seed(n)` sets a position; nothing reads it.

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

## Blocking: features the source assumes exist

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
**Status:** RESOLVED for colour and capability detection. The stateful bar is
still deliberately not adopted; see below.

The resolution was not the one predicted here. Rather than promoting parts of
the terminal layer into `std/`, twill made its `src/term/` and `src/cli/`
modules import each other by a path relative to the importer, and
`resolveImport` tries that path first. So a package that vendors twill under
`twill_modules/` reaches the whole terminal layer, exactly the way weft reaches
`chart` and bobbin reaches `box`. loom now detects capabilities and lights the
loss value and the bar's fill, dropping to plain text the moment the output is
piped.

What loom still does not take is `src/cli/progress.tw`'s determinate bar with
its rate and ETA. That bar is stateful and reads a clock, and wiring it in means
threading a time source through the trainer, which is a separate change with its
own correctness surface. Until then loom keeps its own stateless per-epoch bar,
now lit from the same palette so the two never drift in colour. Duplicating the
progress *logic* is still rejected for the reason it always was.

### 9. `f64` and `i64` conversions, and `F64` as a declared type

**Used by:** `src/metrics.tw` (`update`, `count`), `src/report.tw` (`fixed`),
`src/callback.tw` (`lr_at`, `pack_state`)
**Status:** `i64(f)` and `f64(n)` are specified in section 1.2. `F64` as an
annotation is implied by `Tok.Num(F64)` in the same document and is never
stated.

Systems mode makes annotations mandatory, so a metric value has to be spelled
something. loom writes `F64` and reads it as "a rank-0 tensor with the tensor
machinery off". If the answer is that a scalar in systems mode is still a
`Tensor`, then `Meter.total` is a tensor, every accumulation allocates, and an
epoch of accumulation builds a chain of them; that is a performance answer loom
would want to know about before the interpreter finds out.

### 10. Struct field mutation through a parameter

**Used by:** everywhere. `src/callback.tw` (`observe` writes `c.wait`),
`src/trainer.tw` (`fit` writes `r.st.epoch`), `src/metrics.tw` (`update`)
**Status:** section 1.2 says structs have reference semantics; the systems-mode
rules for assigning to a field of a parameter are not spelled out.

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
**Status:** `Arr[T]` is generic in the design; whether `T` may be `Tensor` is
not stated.

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
**Status:** none. `tests/harness.tw` is a hand-rolled counter and `report` calls
`exit(1)`.

Every test file is a program that has to be run individually, and there is no
way to run the suite except a shell loop, which the CI workflow does not have
because there is nothing to loop over yet. A `twill test` that collected
`*_test.tw`, ran each in a fresh interpreter and reported once would replace
`tests/harness.tw` entirely, and the same file exists in spool, so the
duplication is already two deep.

### 16. Timing, for the progress estimate

**Needs:** a monotonic clock, `mono_ms() -> I64`
**Used by:** `src/report.tw`, which does not have one and therefore reports a
fill with no estimate
**Status:** not in the language; see bobbin's `docs/needs.md`, where it is the
first entry.

The useful part of a progress bar for a 400-epoch run is the time remaining,
not the fill. loom cannot compute it. This is the same requirement bobbin has
and it should be satisfied once.

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
