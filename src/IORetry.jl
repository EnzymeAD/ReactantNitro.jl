# IORetry.jl
#
# Retrying the one I/O path in the framework whose failure loses hours of compute.

"""
    ReactantNitro.with_io_retry(f; attempts, backoff) -> f()

Retry a filesystem operation that failed transiently. A checkpoint write is the one I/O path whose
failure loses hours of compute; writes go through here to a temporary path renamed into place, and
rotation deletes the displaced file only after the new one is durable.
"""
function with_io_retry(f; attempts::Int = 5, backoff::Real = 0.5)
    attempts >= 1 || error("ReactantNitro: `with_io_retry` needs at least one attempt, and got \
                            $attempts.")
    for attempt in 1:attempts
        try
            return f()
        catch err
            # Only transient I/O is retried; a `MethodError` will not succeed on the fourth try.
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

Whether a retry can fix the exception: `SystemError`, `Base.IOError`, and `EOFError` (a truncated
read of a file another process is still writing). Everything else is rethrown on the first attempt.
"""
is_transient_io(err) = err isa SystemError || err isa Base.IOError || err isa EOFError
