# records/

The project's lab notebook, exported as it was written: one folder per experiment, usually with

- `PREREG.md`: the hypothesis, sample size and pass rule, committed **before** the data existed;
- `RESULT.md`: what happened, read by the registered rule (negative and inconclusive results included);
- machine-readable receipts (`*.json`, `*.jsonl.xz`) that the write-ups quote.

Because these are working notes rather than polished prose, read these caveats first.

## What was changed for the public copy

- **Machine details removed.** Process lists captured by the GPU-client checks (which desktop programs were
  running while a benchmark ran) are emptied in the JSON receipts, the lab checkout's absolute path is stripped
  from every file, and a few program names in prose are replaced by generic wording ("a browser", "a desktop
  app"). The write-ups' conclusions about how much outside GPU load there was are unaffected.
- **Digests.** Write-ups cite sha256 digests of files. A JSON receipt that embedded a process list or an absolute
  path was rewritten by the steps above, so its digest in the prose is that of the lab original and will not
  match the public file. Files that were not rewritten (the compressed hand logs, for example) still match.
- **Not everything is here.** Some folders and files that the write-ups cite (`rm2`, `q4-gpu`, scratch
  pre-gates, draft pre-registrations, the 1860.4 rating receipt) were not exported. A few quoted figures therefore
  exist only in a write-up's prose.
- **Cross-references to the lab notebook.** Many files point at `HANDOFF.md`, `FLOPSEARCH.md`, `RATING.md` and
  similar top-level notes of the private working repository. They are condensed into [`../docs/`](../docs) for
  this tree; the originals are not published.

## Who "the user" and "the assistant" are

Written as a working log, the notes say "the user" for the project's author, and the author's go/no-go decisions
are recorded as "the user's yes". The project was developed with an AI coding assistant (Anthropic's Claude), which
appears in the notes as "the assistant" where it carried out or proposed a step (the name is dropped there to keep
the notes readable; the workflow is described in the [top-level README](../README.md#how-it-was-built)); where the assistant, not the author, chose a number or an
operationalisation, the note says so. The decisions to adopt or drop a method were the author's.

## Reading order

[`../RESULTS.md`](../RESULTS.md) summarises everything here; `slumbot-2/RESULT.md` is the most complete single
example of the method.
