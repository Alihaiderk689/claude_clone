# Local Coding Assistant (VS Code extension)

A thin VS Code UI for the local, free, Claude Code-style coding assistant
(the Python `code-agent` project). This extension contains **no agent
logic of its own** -- every response, tool call, file edit, command
execution, and Git operation happens in the Python agent; this extension
only renders what the agent's local HTTP server sends and forwards your
approvals back to it.

This extension is **not published to the VS Code Marketplace**. Install it
locally from a `.vsix` file (see below).

## Prerequisites

1. Python `code-agent` installed and its virtualenv active (see the main
   project's README.md, one directory up).
2. [Ollama](https://ollama.com) running locally with a model pulled, e.g.:
   ```
   ollama serve
   ollama pull qwen2.5-coder:3b
   ```
3. The local agent server running in the folder you want to work on:
   ```
   cd /path/to/your/project
   code-agent serve
   ```
   Leave this running. It binds only to `127.0.0.1` and prints the port it
   picked, the configured model, and Ollama's connection status.

## Install the extension

From this directory:

```
npm install
npm run compile
npx @vscode/vsce package
```

This produces a `.vsix` file. Install it in VS Code with:

```
code --install-extension local-coding-assistant-0.1.0.vsix
```

or via the Extensions view: `...` menu -> "Install from VSIX...".

## Usage

- **Local Coding Assistant: Open Chat** (Command Palette) -- opens the chat
  panel for the current workspace folder.
- Select code, right-click -> **Ask Local Coding Assistant** -- opens the
  chat with the selection pre-filled as context; type your question and
  send.
- **Local Coding Assistant: Explain Selection** / **Fix Selection** --
  same idea, sent automatically.
- File references like `backend/auth.py:42` in the agent's replies are
  clickable and open that file at that line.
- Proposed edits show **View Diff** (opens VS Code's native diff editor)
  plus **Approve**/**Reject**. Proposed commands show **Run**/**Reject**
  with the working directory. Proposed Git operations show
  **Stage**/**Commit**/**Create Branch** plus **Cancel**. Nothing is
  written to disk, executed, or committed until you approve it -- the
  Python agent enforces this the same way it does for the CLI.
- **Stop Task** interrupts the current turn. **New Task** resets the
  conversation and task state for this workspace (same as the CLI's
  `/new`); it does not touch your files or Git history.
- The status indicator (top of the panel) shows ● Connected / ○ Offline
  against the agent server, plus the configured model name -- both read
  from the agent, never hardcoded here.

## Troubleshooting

- **"Agent server unreachable"** -- `code-agent serve` isn't running (or
  isn't running for this project). Start it in the workspace folder.
- **Ollama not connected** (shown next to the model name) -- start Ollama
  with `ollama serve`.
- **Model unavailable** -- pull it: `ollama pull <model-name>`.
- **Port already in use** -- `code-agent serve` automatically falls back
  to an OS-assigned port if 8765 is busy; the extension reads the actual
  port from `~/.code-agent/server.json`, so no action is usually needed.
- **Authentication failure** -- the extension reads its token from
  `~/.code-agent/server.json`, written fresh each time `code-agent serve`
  starts. If you restarted the server, reload the VS Code window so the
  extension picks up the new token.
