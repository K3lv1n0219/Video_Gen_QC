# Project rules

- Keep V1 small; preserve independent full, existing-image, and QC-only entry modes.
- The QC system evaluates observable video quality and task compliance. It does not
  estimate robot executability or use downstream manipulation success as a criterion.
- QC receives the original task and visual evidence, never generated prompts,
  generator self-evaluation, perception outputs, simulation outputs, or rewards.
- Use separate stateless requests for image prompts, video prompts, and QC.
- Judge only inspected frames; use `uncertain` for insufficient evidence. Validate
  evidence IDs and compute the overall decision in Python, never in the VLM.
- Defaults and all tests must run offline. Mock QC must be marked and yield REVIEW.
- Every real provider requires explicit configuration and `--allow-paid`.
  Providers load secrets only from environment variables and never log them.
  The CLI may populate variables from a Git-ignored local .env without overriding
  shell variables. Keep saved keys owner-readable/writable only; never commit them.
- Do not add regeneration loops, robotics, or auxiliary visual tools in V1.
- Verify with `python -m ruff check .`, `python -m ruff format --check .`, and
  `python -m pytest`. Exercise the CLI mock demo after pipeline changes.
- Work on a feature branch; never commit this work directly to main.
