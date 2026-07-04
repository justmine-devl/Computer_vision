# Source Layout

`src/training/` contains main training files.

`src/experiments/` contains experiment, comparison, and analysis scripts. If code is only used once, keep it here.

`src/utils/` contains reusable utilities. If code is reused by two or more scripts, move it here.

Run scripts from the repository root so their path setup can find `src/` and `dl_nets/`.
