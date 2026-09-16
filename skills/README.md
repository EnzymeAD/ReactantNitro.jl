# ReactantNitro skills

Agent skills that teach the framework. They are **public and self-contained**: each states its facts
in full, and none of them references infrastructure outside this repository.

Start at `reactantnitro-experiment`, which indexes the rest.

**Most people who open these are porting a model onto this framework rather than starting one on it,
and that changes what they need.** These pages are written about what this framework does. A porting
reader needs the delta: the places where this framework's default is not the one their model has been
running under. That difference is invisible by construction, because a default is what you get by not
mentioning something, so a port that transcribes only what the source model states explicitly adopts
this framework's answer for everything else, silently and without an error.

So the skills that own a default say so at the point of the default, marked **"If you are porting"**.
Read those lines even when the surrounding section looks like something you already know: the whole
hazard is that the value looks familiar and is not the one you had.

| Skill | Use it when |
| --- | --- |
| `reactantnitro-experiment` | writing, porting, or reading an experiment: the four hooks, the three markers, the batch contract, what a hook may return |
| `reactantnitro-accelerators` | choosing or configuring the accelerator a run executes on (CPU/GPU/TPU), pinning `n_devs`, restricting GPUs with `CUDA_VISIBLE_DEVICES`, or understanding the one-process-one-XLA model |
| `reactantnitro-metrics` | adding, porting, or debugging a metric: `(sum, count)`, the fixed key set, `finalize_metrics`, residency |
| `reactantnitro-optimizer` | parameter groups and per-group learning-rate ratios, `Decay` with its `:zero` and `:w0` anchors (L2 and L2-SP), per-leaf decay exclusion, clipping, schedules, and the binding report |
| `reactantnitro-manual` | manual training mode, where the experiment owns the step: `train_step`, `setup_optimizers`, `backward`, `step_optimizer` |
| `reactantnitro-recompiles` | verifying no recompile, or working out why a REPL edit did nothing |
| `reactantnitro-checkpoint-resume` | configuring checkpoints, resuming a run, debugging a refused resume, early stopping |
| `reactantnitro-device-boundary` | code that passes on CPU and fails on a GPU, and any new residency contract |
| `reactantnitro-visualization` | rendering data or predictions: what to draw, when, the two hooks (`visualize`, `save_figure`), and the `render` driver |
| `reactantnitro-export` | shipping a trained model: the export hooks, the wire seam, what is derived and what you declare |
| `reactantnitro-kaimon` | driving runs from a Kaimon-hosted session: the `nitro_*` tools, the background-run model, naming an experiment, the map from tools to framework verbs |

## Installing

### Pi

Pi discovers skills from `~/.pi/agent/skills/`, a project `.pi/skills/`, and
the `skills` array in `settings.json`. Pick one:

**Per project** (recommended), point pi at this repository's `skills/`
directory from the project's `.pi/settings.json`:

```json
{
  "skills": ["/absolute/path/to/ReactantNitro.jl/skills"]
}
```

**Global**, symlink the skill directory into the agent skills directory:

```bash
ln -s /path/to/ReactantNitro.jl/skills/reactantnitro-experiment ~/.pi/agent/skills/
```

**One run**, load a skill explicitly:

```bash
pi --skill /path/to/ReactantNitro.jl/skills/reactantnitro-experiment "port this training loop"
```

### Claude Code

This repository is a Claude Code plugin marketplace. From a Claude Code
session:

```
/plugin marketplace add EnzymeAD/ReactantNitro.jl
/plugin install reactantnitro-jl
```

The plugin's skills (this `skills/` directory) then load automatically; the
manifest lives in `.claude-plugin/`.

## Harness-agnostic by design

These skills teach the framework, not a workflow or a harness. **None of them assumes that a
Kaimon session exists, and none of them assumes it does not.** `reactantnitro-kaimon` teaches
the tool form of the framework verbs for the harnesses that host one, and a harness-specific
skill downstream (a private package layering workflows on top) may then say "train the model"
and rely on the agent knowing the mechanism: launch with `nitro_train`, poll `nitro_status`,
stop with `nitro_stop`. The general surface stays here so every ReactantNitro user benefits; a
downstream skill adds only the workflow.

## Keeping them true

**A skill that teaches a framework has to version with it.** These live in this repository so that a
change to the surface they describe shows up in the same diff as the change itself. The marker rename
invalidated three of them in a single commit; anywhere else, that goes unnoticed.

So: when you change a public surface, grep `skills/` before you open the pull request. That
includes the tool surface: the `nitro_*` tools are registered by the KaimonGate extension, and a
change to a verb they drive belongs in the same diff as the note in `reactantnitro-kaimon` and in
whatever skill owns that verb.

**Visualization used to be the exception and no longer is.** While the framework shipped no
rendering driver, `reactantnitro-visualization` taught a recipe: the judgement about what to draw
was durable, but the code it showed was expected to collapse into a hook when the `visualize`
surface was built. That surface landed (`visualize`, `save_figure`, `render`), the page now
teaches it, and the exception is gone: every page here is written about a surface.

## Scope

Deliberately not covered yet, because nothing has needed them enough to be sure what to say:
implementing a logger backend, and multi-device. Add one when a real task wants it, rather than in
advance.

The list has shrunk as the framework shipped, which is the rule working as intended: a real task
wanted each of these, so each exists and is written about a surface rather than about a recipe.
Prefetch is a framework default now and lives in `reactantnitro-experiment`'s batch contract; early
stopping lives in `reactantnitro-checkpoint-resume`; manual mode is its own skill,
`reactantnitro-manual`; and export and visualization left the list when their interfaces landed.
