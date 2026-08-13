/**
 * HTTP client for the local Python agent server (agent/server.py).
 *
 * Deliberately has no dependency on the `vscode` module so it can be
 * exercised by plain node:test unit tests without a running VS Code host.
 * Uses only Node built-ins -- no bundled HTTP library -- consistent with
 * the Python side's minimal-dependency approach.
 *
 * This module never re-implements any agent behavior: it only sends HTTP
 * requests to the endpoints agent/server.py exposes and parses their
 * responses (including the chunked NDJSON event stream from /chat). All
 * actual reasoning, tool execution, and approval enforcement happens on
 * the Python side.
 */
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";

export interface ServerInfo {
  host: string;
  port: number;
  token: string;
  pid?: number;
}

export interface HealthResponse {
  status: string;
  backend: string;
  model: string;
  ollama_host: string;
  ollama_connected: boolean;
  workspace?: "ok" | "not_found";
}

export type AgentEvent = { type: string; [key: string]: unknown };

export class AgentUnavailableError extends Error {}
export class AgentAuthError extends Error {}

export function defaultTokenFilePath(): string {
  return path.join(os.homedir(), ".code-agent", "server.json");
}

export function readServerInfo(tokenFilePath: string): ServerInfo {
  let raw: string;
  try {
    raw = fs.readFileSync(tokenFilePath, "utf8");
  } catch {
    throw new AgentUnavailableError(
      `Could not find the local agent's token file at ${tokenFilePath}. Start it with: code-agent serve`
    );
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed.host !== "string" || typeof parsed.port !== "number" || typeof parsed.token !== "string") {
      throw new Error("missing fields");
    }
    return parsed as ServerInfo;
  } catch {
    throw new AgentUnavailableError(`The token file at ${tokenFilePath} is not valid. Restart: code-agent serve`);
  }
}

function parseErrorBody(status: number, body: string): string {
  try {
    const parsed = JSON.parse(body);
    if (parsed && typeof parsed.error === "string") {
      return parsed.error;
    }
  } catch {
    // fall through to the generic message below
  }
  return `The local agent returned an unexpected error (HTTP ${status}).`;
}

/** Splits a stream of arbitrary chunks into complete "\n"-terminated lines. */
export class LineSplitter {
  private buffer = "";

  push(chunk: string): string[] {
    this.buffer += chunk;
    const lines: string[] = [];
    let newlineIndex: number;
    while ((newlineIndex = this.buffer.indexOf("\n")) >= 0) {
      lines.push(this.buffer.slice(0, newlineIndex));
      this.buffer = this.buffer.slice(newlineIndex + 1);
    }
    return lines;
  }
}

export class AgentClient {
  constructor(private readonly tokenFilePath: string = defaultTokenFilePath()) {}

  private info(): ServerInfo {
    return readServerInfo(this.tokenFilePath);
  }

  /** GET /health never requires the token, so a status indicator can
   * distinguish "agent not running" from "agent running, bad token".
   * Passing `workspaceRoot` additionally reports whether that specific
   * folder still exists (e.g. renamed/deleted/unmounted since the panel
   * was opened) -- omit it for a plain "is the agent up at all" check. */
  async health(workspaceRoot?: string): Promise<HealthResponse> {
    const info = this.info();
    const qs = workspaceRoot ? `?workspace_root=${encodeURIComponent(workspaceRoot)}` : "";
    return this.requestJson<HealthResponse>(info, "GET", `/health${qs}`, undefined, false);
  }

  async taskStatus(workspaceRoot: string): Promise<Record<string, unknown>> {
    const info = this.info();
    const qs = `?workspace_root=${encodeURIComponent(workspaceRoot)}`;
    return this.requestJson(info, "GET", `/task/status${qs}`);
  }

  async taskStop(workspaceRoot: string): Promise<Record<string, unknown>> {
    const info = this.info();
    return this.requestJson(info, "POST", "/task/stop", { workspace_root: workspaceRoot });
  }

  async taskNew(workspaceRoot: string): Promise<Record<string, unknown>> {
    const info = this.info();
    return this.requestJson(info, "POST", "/task/new", { workspace_root: workspaceRoot });
  }

  async chatConfirm(workspaceRoot: string, approved: boolean): Promise<Record<string, unknown>> {
    const info = this.info();
    return this.requestJson(info, "POST", "/chat/confirm", {
      workspace_root: workspaceRoot,
      approved,
    });
  }

  /** Streams NDJSON events from POST /chat, calling onEvent for each one
   * as it arrives (progressive rendering, no polling). Resolves once the
   * server closes the response (the turn is over, in whatever way it
   * ended -- final/error/cancelled/max_iterations). */
  chat(workspaceRoot: string, message: string, onEvent: (event: AgentEvent) => void): Promise<void> {
    const info = this.info();
    const body = JSON.stringify({ workspace_root: workspaceRoot, message });

    return new Promise((resolve, reject) => {
      const req = http.request(
        {
          host: info.host,
          port: info.port,
          path: "/chat",
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(body),
            Authorization: `Bearer ${info.token}`,
          },
        },
        (res) => {
          if (res.statusCode === 401) {
            res.resume();
            reject(new AgentAuthError("The local agent rejected this request's authentication token."));
            return;
          }
          if (res.statusCode && res.statusCode >= 400) {
            let errBody = "";
            res.setEncoding("utf8");
            res.on("data", (c) => (errBody += c));
            res.on("end", () => reject(new Error(parseErrorBody(res.statusCode as number, errBody))));
            return;
          }

          const splitter = new LineSplitter();
          res.setEncoding("utf8");
          res.on("data", (chunk: string) => {
            for (const line of splitter.push(chunk)) {
              if (!line.trim()) continue;
              try {
                onEvent(JSON.parse(line) as AgentEvent);
              } catch {
                // One malformed line shouldn't take down the whole stream.
              }
            }
          });
          res.on("end", () => resolve());
          res.on("error", (err) => reject(err));
        }
      );
      req.on("error", (err) => reject(this.translateConnError(err)));
      req.write(body);
      req.end();
    });
  }

  private requestJson<T = Record<string, unknown>>(
    info: ServerInfo,
    method: string,
    requestPath: string,
    body?: unknown,
    requireAuth = true
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const payload = body !== undefined ? JSON.stringify(body) : undefined;
      const headers: Record<string, string> = {};
      if (payload !== undefined) {
        headers["Content-Type"] = "application/json";
        headers["Content-Length"] = String(Buffer.byteLength(payload));
      }
      if (requireAuth) {
        headers["Authorization"] = `Bearer ${info.token}`;
      }

      const req = http.request({ host: info.host, port: info.port, path: requestPath, method, headers }, (res) => {
        let data = "";
        res.setEncoding("utf8");
        res.on("data", (c) => (data += c));
        res.on("end", () => {
          if (res.statusCode === 401) {
            reject(new AgentAuthError("The local agent rejected this request's authentication token."));
            return;
          }
          if (res.statusCode && res.statusCode >= 400) {
            reject(new Error(parseErrorBody(res.statusCode, data)));
            return;
          }
          try {
            resolve(data ? JSON.parse(data) : {});
          } catch {
            reject(new Error("The local agent returned a malformed response."));
          }
        });
      });
      req.on("error", (err) => reject(this.translateConnError(err)));
      if (payload !== undefined) req.write(payload);
      req.end();
    });
  }

  private translateConnError(err: NodeJS.ErrnoException): Error {
    if (err.code === "ECONNREFUSED" || err.code === "ETIMEDOUT" || err.code === "ENOTFOUND") {
      return new AgentUnavailableError('Could not connect to the local agent. Start it with: code-agent serve');
    }
    return err;
  }
}
