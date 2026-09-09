# Security Guidelines

## Secrets

- NEVER hardcode API keys, tokens, passwords, or credentials in any file
- Always use environment variable references: `${VAR_NAME}` or `process.env.VAR_NAME`
- Never echo, log, or print secret values to the terminal

## Permissions

- Never use `--dangerously-skip-permissions` or `--no-verify`
- Do not run `sudo` commands
- Do not use `rm -rf` without explicit user confirmation
- Do not use `chmod 777` on any file or directory

## Running commands

- Any Bash/shell command that runs Python must be handed off to the user to
  run themselves — do not execute it. This covers `python`, `python3`, `py`,
  `pip`, `python -m ...`, running any `.py` script, and Python-based runners
  like `flask`, `gunicorn`, and `pytest`.
- To hand off, give the user the exact command and ask them to run it with the
  `! <command>` prefix in the prompt, then wait for their pasted output before
  continuing.
- Non-Python commands (`git`, `ls`, `npm`, etc.) may still be run normally.

## Code Safety

- Validate all user inputs before processing
- Use parameterized queries for database operations
- Sanitize HTML output to prevent XSS
- Never execute dynamically constructed shell commands with user input

## MCP Servers

- Only connect to trusted, verified MCP servers
- Review MCP server permissions before enabling
- Do not pass secrets as command-line arguments to MCP servers
- Use environment variables for MCP server credentials

## Hooks

- All hooks must be reviewed before activation
- Hooks should not exfiltrate data or make external network calls
- PostToolUse hooks should validate output, not modify it silently

## During changes

- When making a change, do not break existing features built on top of the affected code. E.g. a frontend change must not break backend functionality.
- Keep the frontend and backend decoupled: changing the frontend for design/visual reasons must not change app behavior. All functionality lives in the backend.
- The frontend is for visual presentation only — it does not call external APIs, hold system prompts for the AI, or contain business logic. Its only job is to display data; everything else happens in the backend.

## File access scope

- Do NOT create, edit, move, rename, or delete any file inside `public/` or `lib/`.
  These two directories are off-limits for writes.
- NEVER delete, move, or rename any file inside `public/` under any circumstances.
  Vercel deploys with `public/` as its Root Directory, so it is the entire
  deployed site -- a removed file breaks production. Any fix to Vercel or
  frontend behaviour must be made by EDITING files within `public/`, never by
  removing them. (Deletion stays forbidden even where an edit is authorized
  below.)
- EXCEPTION (OpenAI Agents SDK migration + single-ticker fetch rebuild + trace
  grouping + retrospective "thinking" step log + Vercel static-deploy fix,
  authorized by the repo owner 2026-09-09): EDITING these files is permitted
  (deletion of `public/` files is still forbidden), and only for the migration
  specified in `docs/agents-sdk-migration/`, the single-ticker rebuild of
  `fetch.py`, the `trace()` wrapper in `pipeline.py`, the per-step timing log
  (`steps` array in the `/api/ask` response + its frontend panel), and keeping
  Vercel serving `public/` as a static site (no Flask detection):
  - `lib/yahoo_finance/config.py`
  - `lib/yahoo_finance/intent.py`
  - `lib/yahoo_finance/answer.py`
  - `lib/yahoo_finance/fetch.py`
  - `lib/yahoo_finance/pipeline.py`
  - `public/js/ask.js`
  - `public/js/render.js`
  - `public/style.css`
  - `public/vercel.json`
  - `public/pyproject.toml`
  All other files under `lib/`, and all of `public/`, remain off-limits for writes.
- Every other file and directory in the repo is in scope. You may read and modify
  `api/`, root config files (`vercel.json`, `.vercelignore`, `pyproject.toml`,
  `requirements.txt`, `render.yaml`, ...), and anything else outside `public/` and `lib/`.
- Reading files inside `public/` and `lib/` is allowed; only writing to them is not.
- If a task requires changing a file inside `public/` or `lib/`, stop and explain what
  is needed rather than editing it.