# ReactantNitro.jl conventions

## Formatting: Runic.jl

**Format Julia source with [Runic.jl](https://github.com/fredrikekre/Runic.jl) before committing.**
Runic is deliberately non-configurable, so there is nothing to agree about and no config file to
drift: what it emits is the format.

```bash
runic src ext test docs  # format in place
runic --check --diff src ext test docs  # verify without writing, and show what it would change
```

A pre-commit hook runs `runic --inplace src ext test docs` and re-stages the formatted files, so
commits never ship unformatted code. `.git/hooks/` is machine-local and not versioned; (re)install
the hook in a fresh clone with:

```bash
cp scripts/pre-commit-runic .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

`runic` is a standalone tool and **not a dependency of this project**; it is invoked as `julia -m
Runic` when the CLI shim is not on `PATH`. Do not add it to `Project.toml`.

Two things it does not do, both of which still matter here:

- **It does not reflow comments or docstrings**, which is most of this codebase. The 100-column
  convention for prose is yours to keep.
- **It does not touch the `─` box-drawing section headers** used throughout `src/`. Keep them.

Run it over the files you touched rather than the whole tree in an unrelated change, so a diff stays
about its subject.

## Prose

**No em dashes or en dashes anywhere**, in code, comments, docstrings, documents, or commit
messages. Use a comma, a colon, a semicolon, or restructure the sentence.

Commit messages carry the reasoning, not just the change.
