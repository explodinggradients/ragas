Hey @JohnCarl-30, thanks for reporting this issue! I can take this up and submit a PR to resolve it.

Here is how I plan to implement the fix:
1. **Format the files:** I'll run `make format` (or `uv run ruff format docs src tests --config pyproject.toml`) to reformat the 88 `.md` files and ensure everything is consistent with the current `ruff` version.
2. **Pin the `ruff` version:** I'll update `pyproject.toml` to explicitly pin `ruff` in the development dependencies (e.g. `ruff == 0.16.2`). This will help reduce future drift and ensure that all contributors are using the same version.
3. **Open a PR:** I'll commit these changes and open a Pull Request so that the pre-commit hook runs smoothly for all fresh checkouts.

I'll get started on this and have a PR ready shortly! Let me know if you have any additional thoughts.
