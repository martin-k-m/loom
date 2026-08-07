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

**Used by:** `src/trainer.tw` (`fit` returns a run and an error),
`src/checkpoint.tw` (`restore`), `src/callback.tw` (`validate`,
`unpack_state`), `src/data.tw` (`take`, `split`)
**Status:** `Res[T, E]` and `Opt[T]` are in section 1.2 of the self-hosting
design and need generics; multiple returns are not designed anywhere.

Every fallible function in loom returns a `Str` that is empty on success, which
is spool's convention and has spool's problem: the compiler does not make anyone
read it. Every function that wants to return two things returns a struct
declared for that one call site, which is why `src/data.tw` has a `Batch` and
`src/state.tw` has a `StepResult`. Neither struct is a type anyone wanted; both
are a tuple with a name.

### 5. Sum types and `match`

**Used by:** `src/callback.tw` (`Callback.kind`, `fire`), `src/state.tw`
(`OptState.kind`), `src/callback.tw` (`Callback.sched`)
**Status:** designed in section 1.2 of the self-hosting design, not implemented.

`Callback` is one flat struct holding the union of every callback's fields with
an I64 discriminant, and `fire` is an if-chain over it. Three discriminants in
this repository are I64 constants: callback kind, optimiser kind, schedule
shape. Every one of them wants to be an enum, and every one of them has the same
failure mode: adding a variant compiles and silently does nothing, because
nothing forces the if-chain to grow an arm. Exhaustive `match` is the feature
that turns adding a callback from a hunt into a compile error, and a callback
framework is exactly the kind of code that gets extended by people who did not
write it.

The workaround also costs correctness, not only tidiness: a checkpoint callback
carries `patience` and `min_delta` fields it never reads, and nothing stops a
caller setting them and expecting an effect.

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
overloading them on I64 seems worse than a function. It also assumes `shr` is a
logical shift, which matters: splitmix64's finaliser is wrong with an arithmetic
shift, and every seed derived from a negative base seed would be wrong with it.
This is the one place in loom where the wrong answer is silent.

### 8. twill's terminal layer, reachable from a package

**Needs:** `src/term/` and `src/cli/` available as `std/` modules, or an
equivalent
**Used by:** `src/report.tw`, which would call them and does not
**Status:** they exist, in the twill repository, as files.

twill resolves a non-`std/` import as a path relative to the importing file, so
only modules under `std/` are reachable from an installed package. `src/term/`
is not one. loom therefore has no colour, no capability detection, no width
handling, and no progress bar, and prints one plain line per epoch instead.

The right fix is not to widen the import rule. It is to promote the parts of the
terminal layer that a library legitimately needs into `std/term`: capability
detection, the palette, and the determinate bar. loom will delete
`src/report.tw`'s formatting helpers the day that exists. Duplicating
`src/cli/progress.tw` here was the obvious alternative and was rejected: two
progress bars in one ecosystem drift, and the drift is visible to users.

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
**Status:** milestone 1 gives `Dict[Str, V]`; whether `V` may be a struct is
not stated.

`MeterSet` is two parallel `Dict[Str, F64]` plus an `Arr[Str]` of names, where
it wants to be one `Dict[Str, Meter]`. The parallel dicts can go out of step and
only a convention keeps them together.

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
