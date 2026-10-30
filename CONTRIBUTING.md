# Contributing

Thanks for helping. A few ground rules keep the results trustworthy.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # tests need no models; add ".[models]" to run the real pipeline
ruff check . && ruff format --check . && pytest
```
Or `make test` / `make lint`.

## Rules
1. **`paper` must not change.** It reproduces the published method. Improvements go behind a new preset or an
   opt-in config flag and need a test and a line in the README comparison.
2. **Never fabricate numbers.** Every figure in the docs must come from a script in this repo. If you could not run
   something, say "not run" and why.
3. **No copyrighted media and no secrets.** Do not commit videos, audio, model weights, `.env` or tokens. Test with
   synthetic audio or openly licensed clips and state the licence in `docs/DATA.md`.
4. **Mask profanity in docs** (`f***`). Only locally produced JSON/CSV may contain the real words.
5. **Tests must run offline** with no GPU and no model downloads (use the fake transcriber/diarizer). Mark anything
   that needs real models with `@pytest.mark.models`.
6. Keep Python 3.9 compatible (`Optional[X]`, not `X | None`, because pydantic and typer evaluate annotations at runtime).

## Pull requests
Small, focused, Conventional Commit messages (`feat:`, `fix:`, `test:`, `docs:`). Describe what you measured.
