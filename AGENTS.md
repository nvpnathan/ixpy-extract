# UiPath SDK Agent Guide

This guide is the default reference for building new UiPath coded agents in this repo.

## Documentation Structure

This documentation is split into multiple files for efficient context loading. Load only the files you need:

1. **@.agent/REQUIRED_STRUCTURE.md** - Agent structure patterns and templates
   - **When to load:** Creating a new agent or understanding required patterns
   - **Contains:** Required Pydantic models (Input, Output), SDK initialization patterns, standard agent template

2. **@.agent/SDK_REFERENCE.md** - Complete SDK API reference
   - **When to load:** Calling UiPath SDK methods, working with services (actions, assets, jobs, etc.)
   - **Contains:** All SDK services and methods with full signatures and type annotations

3. **@.agent/CLI_REFERENCE.md** - CLI commands documentation
   - **When to load:** Working with `uipath auth`, `uipath run`, or `uipath eval` commands
   - **Contains:** Command syntax, options, usage examples, and workflows

## Required Structure
- Every agent file must define `Input` and `Output` Pydantic models.
- The main entrypoint must be a function named `main`, `run`, or `execute`.
- Initialize the SDK with `uipath = UiPath()` unless you need explicit credentials.
  - Prefer initializing inside `main` to avoid `uipath init` failing when auth/env vars aren’t set.

## Do's
- Do keep `Input`/`Output` models explicit and documented (use `Field` descriptions).
- Do validate inputs with `@model_validator` when fields are conditional.
- Do keep payload mapping logic in a dedicated class or helper functions.
- Do batch Data Service writes (insert/update) to reduce API calls.
- Do surface failures clearly in the `Output` model.
- Do use environment variables for secrets and entity keys.

## Don'ts
- Don’t hardcode tokens, base URLs, or entity keys in code.
- Don’t rely on implicit JSON shapes; normalize input before processing.
- Don’t mutate SDK response objects in-place if you later use them for updates.
- Don’t swallow exceptions from the UiPath SDK without returning a useful error.
- Don’t update fields in Data Service that don’t exist in the entity schema.

## How To Use
1) Define `Input` and `Output` models in `main.py`.
2) Implement the agent logic in `main(input_data: Input) -> Output`.
3) If needed, normalize raw JSON payloads into the `Input` model.
4) Return a structured `Output` with success and error details.

## Files To Modify For Input/Output Arguments
- `main.py`: update the `Input` and `Output` Pydantic models.
- `entry-points.json`: update the `input` and `output` JSON schemas.
- `uipath.json`: mirror the same `input` and `output` schemas and ensure the entry point is listed.
  - If using the top-level `entryPoints` format, copy the generated schema from `entry-points.json`.

## Environment Variables
- `UIPATH_URL` and auth credentials are required by the SDK.
- Use `.env` for local development, and `uipath auth` to refresh credentials.

## Testing
Local:
- `python3 -m py_compile main.py`
- `uipath auth`
- `uipath run <entrypoint> -f <input.json>`
  - `<entrypoint>` must match the key in `uipath.json` under `functions`.
