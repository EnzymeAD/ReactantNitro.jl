# The workload: what a test item does with the shared type BEFORE it executes a compiled program,
# once, at precompile time.
#
# ONE LIMIT SHAPES IT, and it is a hard one. A Reactant thunk compiled during precompilation must
# never be called: the call's generated method, with the precompile process's executable pointers
# baked in, lands in this image under a gensym name that the next process reuses for its own
# thunk of the same program, and every later `train!` in that process then fails at
# `XLAExecuteSharded` with a `MethodError` on a doubly wrapped `Ref`. Measured, twice, from the
# test workers. Reactant's own workload compiles and never executes for this reason, and this
# framework's `train!`, `validate` and `predict` compile and execute in one motion, so none of them
# can be here.
#
# What is left is still most of what a construction-heavy item pays, and a real share of a
# training item: the setup sequence and its checks, the layout, routing, the schedule resolution,
# the logger, the report and its table, the history data and table, the checkpoint search, and the
# warm-start transfer, all inferred for `KitMLP` and cached in this image. The trace and the
# training loop stay cold; warming those needs a compile-without-execute entry point in the
# framework, which does not exist yet.
#
# Each stage is fenced on its own, so a stage that fails does not skip the ones after it, and the
# workload reports rather than failing precompilation: a worker that could not warm still runs
# every test, only cold.
using PrecompileTools: @setup_workload, @compile_workload

function _stage(f, stopped, name)
    try
        return f()
    catch err
        # The first line of the error rides along, so the warning below says what broke rather
        # than only where.
        push!(stopped, name * " (" * first(split(sprint(showerror, err), '\n')) * ")")
        return nothing
    end
end

function _kit_workload()
    stopped = String[]
    setup_devices!(backend = "cpu")
    dir = mktempdir()
    ckpt() = TopKCheckpointer(; metric = :mae, mode = :min)
    try
        # Construction: the setup sequence, the layout, routing, schedules, the logger, the report.
        n = _stage(stopped, "construct") do
            Nitro(KitMLP(; max_epochs = 2); run_dir = dir, checkpointer = ckpt())
        end
        n === nothing && return stopped
        _stage(stopped, "display") do
            sprint(show, MIME"text/plain"(), n)
            sprint(show, n)
            ReactantNitro.build_binding_report(n)
        end
        # The history and its table, on a row shaped exactly as the loop appends it.
        _stage(stopped, "history") do
            # The handle has not trained, so its epoch is 0; the row is stamped as epoch 1 so the
            # epoch-indexed selections below have something to find.
            isempty(n.history) && push!(
                n.history,
                merge(
                    ReactantNitro.history_row(n, 1.0, 1, (; mae = 0.5f0, cm = ones(Int, 2, 2))),
                    (; epoch = 1, step = 4)
                )
            )
            h = history(n)
            sprint(show, MIME"text/plain"(), h)
            sprint(show, h)
            h[1]
            h[1:1, :mae]
            h.mae
            ReactantNitro.history_table(h, 24, 80)
            ReactantNitro.thin_rows(40, 23, 12)
        end
        # Resume: the checkpoint search and the record path, with and without a record to find.
        _stage(stopped, "resume") do
            Nitro(
                KitMLP(; max_epochs = 3); run_dir = dir, data = n.data, resume = :auto,
                checkpointer = ckpt()
            )
        end
        # The warm start: the transfer, the structural check, and each `w0` form.
        _stage(stopped, "weights") do
            n3 = Nitro(
                KitMLP(); run_dir = mktempdir(), data = n.data, weights = n, w0 = :weights,
                checkpointer = nothing
            )
            Nitro(
                KitMLP(); run_dir = mktempdir(), data = n.data, weights = n3,
                w0 = ReactantNitro.to_host(n3.ps), checkpointer = nothing
            )
            sprint(show, MIME"text/plain"(), n3)
        end
    finally
        ReactantNitro.cache_reset!()
        rm(dir; recursive = true, force = true)
        GC.gc()
    end
    return stopped
end

@setup_workload begin
    @compile_workload begin
        stopped = try
            _kit_workload()
        catch err
            @warn "NitroTestKit: the precompile workload failed; the tests will run cold" exception = (err, catch_backtrace())
            String[]
        end
        # Nothing here executes a program, so a stopped stage is unexpected and worth a look.
        isempty(stopped) || @warn "NitroTestKit: workload stages that failed, so the tests run \
                                   cold for them: $(join(stopped, ", "))"
    end
end
