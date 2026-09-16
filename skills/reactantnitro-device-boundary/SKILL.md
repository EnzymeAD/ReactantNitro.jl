---
name: reactantnitro-device-boundary
description: >
  Debug and prevent CPU/GPU divergence in ReactantNitro: why a fully green CPU test suite
  can pass while code has never once worked on a device, what a model author must and must
  not do at the boundary, what scalar indexing does differently on each backend, how to
  read a residency assertion, and the rule to apply to any new contract ("would this raise
  on CPU?"). Invoke when code that passes on CPU fails on a GPU; when anything raises
  "Scalar indexing is disallowed" or "still holds DEVICE-resident values"; when a resume
  dies on `buffer.buffer !== C_NULL`; or when adding a contract that promises host or
  device residency.
---

# The host/device boundary

## The fact that makes this its own skill

**On a CPU backend, a "device" array is host memory, and scalar indexing into one is legal rather
than fatal.** So the host/device distinction is very nearly unobservable there. A test suite can be
fully green while the code has never once done the thing it claims.

This is not hypothetical. The first GPU run of this framework, against a green 1243-test CPU suite,
found six defects in one sitting, **five of them invisible to that suite for this one reason**. Two
made GPU execution impossible outright. They were not five coincidences; they were one cause with
five symptoms.

**The consequence is a rule about reading results, not a list of tests to add.** Do not read a green
CPU run as evidence about device behaviour, and do not expect more CPU tests to change that: the
tests cannot observe the distinction their subject is about.

## If you are writing a model, this is your whole share of it

Three rules. Everything below them is why they are the rules.

**1. Do not convert in a hook.** A `:host` hook receives host arrays, both the outputs and the routed
batch fields, and the framework asserts the conversion was total before calling you. You do not need
`Array(...)` and you should not write it. A defensive conversion makes the hook pass whether or not
the boundary holds, which removes the only signal that it stopped holding, and it costs a transfer
you already paid for.

**2. When an assertion fires, report it rather than working around it.** These are the framework's
own assertions about its own conversions, so one firing means the framework has a gap, and the
message says so. The exception it calls out separately is a value **your own hook returned**, which
the framework never saw before you built it and therefore cannot convert for you.

**3. Do a single-epoch run on the real device before a long one.** Two of this framework's GPU-only
defects were in setup and in the first optimizer step, so a short run finds them in minutes instead
of hours, and the ones that survive to epoch two are usually not of this class at all. Schedule it
early rather than as the last gate: a green CPU suite is not a reason to defer it, it is the reason
you need it.

## Reading the failures

**`Scalar indexing is disallowed` on a `ConcretePJRTArray`.** Something indexed a device array one
element at a time. Common sources: `vec` producing a reshaped view whose `vcat` falls into a generic
element-wise path; `argmax`, `findmax`, or a loop over indices in a hook that expected host arrays;
any Base fallback that has no device implementation. Note that the *generic* path is often legal and
merely slow on host, which is why nothing complained until now.

**`still holds DEVICE-resident values`, at a hook boundary or a checkpoint write.** These are the
framework's own assertions. They name the crossing and the path to the offending leaf, for example
`.st.dps.dropout.rng :: ReactantRNG{ConcretePJRTArray}`. Read the path first: it tells you which
value, not just that there is one.

**`AssertionError: buffer.buffer !== C_NULL`, in a fresh process, from a read-back that mentions
nothing you recognize.** This is the same class one step later: a device array reached a serialized
file, which accepts and returns it without complaint because the pointer inside it is written and
read like any other field. The pointer means nothing in the reading process. If you get this, the
value went to disk in an earlier run, so look at what wrote the checkpoint rather than at what is
reading it. The write-time assertion above exists to convert this into a legible error; seeing this
form instead means the value took a path that assertion does not cover, which is worth reporting.

**These assertions mean the framework's conversion has a gap, not that you forgot `Array(...)`.** The
package handles the device-to-host transfer; that is a standing design commitment, because a contract
where omitting a conversion is silently correct on CPU and fatal on GPU is the worst available failure
gradient for a codebase whose tests are all CPU.

## The rule for anything new

This one is for whoever adds a contract, in the framework or in a model.

**Ask of every contract you add: would violating this raise on CPU?**

- **Yes** → an ordinary test covers it. Write the test.
- **No** → the contract needs **either an assertion or a GPU test**, because the suite cannot speak
  to it. Shipping it with neither means shipping a claim nothing checks.

**Prefer the assertion where both would work**, because an assertion runs in every run anyone ever
does, including on hardware nobody tested. A GPU test only runs where there is a GPU.

The assertion has to fire **at the point of the mistake**, naming what is wrong. That is the entire
difference between a defect that costs ten minutes and one that costs a day: every instance of this
class that was caught by an assertion was diagnosed immediately, and every instance that was not
surfaced hours later, in a different process, from a stack trace naming a function in Base.

## What a CPU test *can* establish here

The conversions and assertions key on the **type** of a leaf, not on the behaviour of indexing it,
and `ConcretePJRTArray` is not `Array` whatever the memory underneath. So the guards themselves are
CPU-testable, and should be tested:

- an assertion returns its argument unchanged for a host tree and raises **naming the path** for a
  device leaf nested inside one
- a conversion reaches inside a struct, a named tuple, and a nested combination of both
- identity preservation: a subtree with nothing device-resident comes back `===` what it was

**A CPU test can check the guard. It can never check the thing the guard protects against.** Keep
that distinction in the test names, so nobody later reads guard coverage as device coverage.

## The worked example, because the shape recurs

Below here is why the rules above are the rules. Both stories are about traversal code inside the
framework; read them if you are writing any, and skip them if you are writing a model.

The framework had **two functions that both claimed to read a tree back to host, and one of them
could not see inside a struct.**

One walked any struct through a generic fallback. The other used a functor-style map, which descends
tuples, named tuples, and arrays but treats an unregistered struct as a **leaf** and left it
unconverted. A device array held inside a plain wrapper struct therefore came back device-resident,
silently, on the path feeding every host-residency metric hook.

Both had been written to do the same job. One had been fixed after this exact class of bug was found
in serialized layer state; the second was never touched, because nothing pointed at it.

**What pointed at it was building the assertion.** The assertion walks *any* struct, so it is
strictly more thorough than the conversion it guards, and reading the two side by side to establish
that made the gap visible. That inequality is not untidiness to clean up: **a backstop that can only
see what the mechanism can see is not a backstop.**

Two transferable lessons:

- When you fix a walker, **grep for the other walkers**. This class recurs in traversal code
  specifically, because "walks containers" and "walks structs" look identical until a struct shows up.
- Keep the checker broader than the thing it checks. If they have the same reach, the checker is
  decoration.

## The blind spot underneath that one

The same framework then hit it a third time, and the third time is the one worth internalizing:
**the recurring blind spot is WRAPPERS AROUND device arrays, not device arrays.**

`view`, `reshape`, and `vec` of a device array produce a `SubArray` or a `ReshapedArray`. Neither is
a device-array type, so any check shaped like "is this a device array" answers **no** and stops:

```julia
device_paths(d, "x")                 # found
device_paths(view(d, :, 1:2), "x")   # MISSED
```

The converter had it from the other side, returning a `SubArray{Float32,2,<device array>}` where the
hook expected a host `Matrix`. So the hook got a device array, **the assertion passed on it**, and
the failure landed in `argmax` one line later.

**And this is where the previous lesson gets a correction worth having.** The checker being broader
than its subject was true for structs and false here: both walkers shared the wrapper assumption, so
the backstop had exactly the blind spot of the thing it was backing up. **A backstop that shares an
implementation assumption with its subject is not independent of it.** Being broader in one dimension
buys nothing in another.

Two things follow for anyone writing this kind of traversal:

- **Ask what the array IS, not only what it holds.** `parent(x) !== x` is the whole test for a
  wrapper, and it is cheap.
- **Watch for two predicates that look like one question and are two.** Here, "is this one array to
  slice" and "could device memory hide under this" give opposite answers for a view, and a slicing
  helper genuinely needs the first. Sharing one predicate between them is the bug; sharing one
  predicate between the two *residency* walkers is the fix.
