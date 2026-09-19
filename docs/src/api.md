# API reference

Every exported name that carries a docstring appears below, collected automatically. The docstrings
are the contract, and the guide pages give the worked context: [Tutorial](tutorial.md),
[Experiments](experiments.md), [Recompilation](recompilation.md),
[Optimization](optimization.md), and [Schedules](schedules.md).

```@autodocs
Modules = [ReactantNitro]
Public = true
Private = false
```

## Compile cache internals

`cache_stats` and `cache_reset!` are deliberately unexported (the guides reach for them from the
REPL as `ReactantNitro.cache_stats`), and they are documented here so their refs resolve.

```@docs
ReactantNitro.cache_stats
ReactantNitro.cache_reset!
```

## Internals referenced by the docstrings

Several docstrings of the public surface point at internal helpers that carry their own docstrings.
They are collected here so those links resolve; they are implementation details and may change
without a breaking release.

```@docs
ReactantNitro.MetricHistory
ReactantNitro.history_table
ReactantNitro.history_series
ReactantNitro.thin_rows
ReactantNitro.StrippedHost
ReactantNitro.publish_phase
ReactantNitro.work_in_flight
ReactantNitro.set_phase!
ReactantNitro.graphconst_field_hash
ReactantNitro.assert_graphconst_hashable
ReactantNitro.check_config_compatible
ReactantNitro._stripped_error
ReactantNitro.check_control_readback
ReactantNitro.with_io_retry
ReactantNitro.PrefetchStream
ReactantNitro.auto_prefetch
ReactantNitro.batch_stream
ReactantNitro.eval_stream
ReactantNitro.check_batch_at
ReactantNitro.close_stream!
ReactantNitro.fanout_capable
ReactantNitro.strip_rules
ReactantNitro.merge_rules
```
