# IORetry.jl
#
# Retrying the one I/O path in the framework whose failure loses hours of compute.

"""
    ReactantNitro.with_io_retry(f; attempts::Int = 5, backoff::Real = 0.5) -> f()

Call `f`, retrying up to `attempts` times in total when it throws a transient I/O error (see
`is_transient_io`). The wait before retry `k` is `backoff * 2^(k - 1)` seconds, and each retry logs
a warning. Any other error, and the error from the last attempt, is rethrown. Checkpoint reads,
writes and rotation deletes go through it, since a failed checkpoint write can lose hours of
training; a write goes to a temporary path that is then renamed into place.
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

Whether [`with_io_retry`](@ref) retries `err`: true for `SystemError`, `Base.IOError` and
`EOFError` (a truncated read of a file another process is still writing). Any other error is
rethrown on the first attempt.
"""
is_transient_io(err) = err isa SystemError || err isa Base.IOError || err isa EOFError
