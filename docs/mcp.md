# Using discopt from Claude Desktop (MCP server)

`discopt mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io)
server on stdio. Any MCP client can then build and solve discopt models on your
machine: Claude Desktop, Claude Code, or another one.

## Install and register

```bash
pip install 'discopt[mcp]'
discopt mcp --print-config
```

The second command prints the entry to merge into Claude Desktop's
`claude_desktop_config.json`. On macOS that file is
`~/Library/Application Support/Claude/claude_desktop_config.json`; on Windows it
is `%APPDATA%\Claude\claude_desktop_config.json`. The entry looks like this:

```json
{
  "mcpServers": {
    "discopt": {
      "command": "/path/to/your/python",
      "args": ["-m", "discopt.cli", "mcp"]
    }
  }
}
```

It names the absolute path of the Python that has discopt installed. Desktop
does not inherit your shell's `PATH`, so a bare `discopt` command is often not
found there. Restart Desktop after editing the file.

For Claude Code, register the same command:

```bash
claude mcp add discopt -- "$(python -c 'import sys; print(sys.executable)')" -m discopt.cli mcp
```

## Tools

| Tool | What it does |
|---|---|
| `solve_python` | Runs Python source that builds a `Model` named `model` or `m`, then solves it |
| `solve_file` | Solves a `.nl`, `.gms`, saved model document (`.dopt`, `.json`, `.json.gz`) or a `.py` script |
| `model_summary` | Describes a model's size and structure without solving it |
| `convert_model` | Converts between `.gms`, `.nl`, `.mps` and `.lp` |
| `discopt_info` | Reports the version, the install location and the Python running the server |

A solve returns the same fields as `discopt solve --format json`: `status`,
`objective`, `bound`, `gap`, `gap_certified`, `wall_time`, `node_count` and the
solution `x`. `x` is truncated to `max_vars` entries (default 200), and when that
happens `x_truncated` says how many entries there were. `time_limit` defaults
to 60 s per call. A result is a proven global optimum only when `status` is
`optimal` and `gap_certified` is true.

## How it runs

Each request runs in a fresh worker process, never in the server process.
There are three reasons:

- **Clean protocol stream.** The stdio transport is the server's stdout, so
  anything a solver or a model script prints there would corrupt it. The worker
  redirects its own stdout to stderr and returns its result on a private channel.
- **Hard timeout.** A worker still running 60 s past its `time_limit` is killed,
  and the call returns an error instead of hanging the client.
- **Crash isolation.** A crash takes down the worker, not the server.

Errors are returned as `{"ok": false, "error": ..., "stderr_tail": ...}`, with
the tail of the worker's traceback.

```{warning}
`solve_python` and `.py` files **execute arbitrary Python** as the user running
the server. That is how a client writes a model, and it is no more than a
client with a shell tool could do. Register the server only with clients you
would trust with a shell.
```
