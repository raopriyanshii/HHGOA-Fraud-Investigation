# Raw data manifest

This folder intentionally does not contain a copy of the organizer files.
`transactions.csv` alone is ~675 MB, and Section 1 of the task rules
requires the organizer-provided files to stay untouched at their original
location. Copying them here would create a second copy that could drift
out of sync and would not add any safety the original doesn't already have
(we only ever open these files read-only).

**Canonical, immutable source of truth:**

```
D:\hackathon\task_5\
    README.md
    transactions.csv
    identity.csv
    closed_cases_history.csv
    case_pack.csv
```

All code in `src/` reads from this path (via a single configurable
constant, not hard-coded in multiple places) and never writes to it.

File sizes and row counts observed at inspection time (2026-09-19), for
drift detection if the folder is ever refreshed:

| File | Bytes | Rows (excl. header) |
|---|---|---|
| transactions.csv | 707,936,515 | 590,742 |
| identity.csv | 26,716,154 | 144,432 |
| closed_cases_history.csv | 2,706,417 | 5,565 |
| case_pack.csv | 3,548 | 20 |

Anything this pipeline derives from the raw files (resolved card
identities, validation reports, GSQL-ready load files, etc.) goes in
`data/processed/`, never here.
