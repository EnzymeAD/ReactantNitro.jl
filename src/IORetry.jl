# IORetry.jl
#
# Retrying the one I/O path in the framework whose failure loses hours of compute.

"""
    ReactantNitro.with_io_retry(f; attempts, backoff) -> f()

Retry a filesystem operation that failed transiently. This exists because a checkpoint write is the
one I/O path in the framework whose failure loses hours of compute.

Checkpoint writes go through here, **write to a temporary path and `rename` into place atomically**,
and **top-K rotation deletes the displaced file only after the new one is durably in place**. A
rotation that deletes first and then fails to write leaves a run with fewer checkpoints than its
retention policy promises.
"""
function with_io_retry(f; attempts::Int = 5, backoff::Real = 0.5)
    attempts >= 1 || error("ReactantNitro: `with_io_retry` needs at least one attempt, and got \
                            $attempts.")
    for attempt in 1:attempts
        try
            return f()
        catch err
            # ONLY TRANSIENT I/O IS RETRIED. A `MethodError` or a serialization failure is not going
            # to succeed on the fourth try, and retrying it would turn an instant, legible failure
            # into a slow one with the original cause four backoffs back in the log.
            (attempt == attempts || !is_transient_io(err)) && rethrow()
            @warn "ReactantNitro: a filesystem operation failed and will be retried \
                   ($attempt/$attempts). A checkpoint write is the one I/O path in the framework \
                   whose failure loses hours of compute." exception = err
            sleep(backoff * 2^(attempt - 1))
        end
    end
    return
end

"""
    ReactantNitro.is_transient_io(err) -> Bool

Whether an exception is the kind a retry can fix. `SystemError` and `Base.IOError` cover the cases
this is about: a full or briefly unavailable filesystem, a network-filesystem hiccup, a stale
handle. `EOFError` is included because a truncated read of a file another process is still writing
is the same class of fault.

Everything else is rethrown on the first attempt. Retrying a `MethodError`, a `JLD2` type error, or
a `Base.InvalidStateException` cannot help, and it hides the cause behind several seconds of
backoff.
"""
is_transient_io(err) = err isa SystemError || err isa Base.IOError || err isa EOFError
