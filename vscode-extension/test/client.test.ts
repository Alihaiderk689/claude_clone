/**
 * Tests for the pure/testable parts of client.ts: NDJSON line splitting,
 * token-file parsing, and the HTTP request/response handling against a
 * throwaway local HTTP server that stands in for agent/server.py. Run with
 * `npm test` (node:test against the compiled output in out/), no VS Code
 * host required -- see the module docstring in src/client.ts for why this
 * file has no dependency on the `vscode` module.
 */
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, test } from "node:test";

import {
  AgentAuthError,
  AgentClient,
  AgentUnavailableError,
  LineSplitter,
  readServerInfo,
} from "../src/client";

describe("LineSplitter", () => {
  test("returns nothing until a newline arrives", () => {
    const splitter = new LineSplitter();
    assert.deepEqual(splitter.push("partial"), []);
  });

  test("splits a single chunk with multiple lines", () => {
    const splitter = new LineSplitter();
    assert.deepEqual(splitter.push('{"a":1}\n{"b":2}\n'), ['{"a":1}', '{"b":2}']);
  });

  test("reassembles a line split across chunks", () => {
    const splitter = new LineSplitter();
    assert.deepEqual(splitter.push('{"a":'), []);
    assert.deepEqual(splitter.push('1}\n'), ['{"a":1}']);
  });

  test("keeps a trailing partial line buffered across pushes", () => {
    const splitter = new LineSplitter();
    assert.deepEqual(splitter.push("one\ntwo\nthr"), ["one", "two"]);
    assert.deepEqual(splitter.push("ee\n"), ["three"]);
  });
});

describe("readServerInfo", () => {
  test("throws AgentUnavailableError when the token file does not exist", () => {
    const missing = path.join(os.tmpdir(), "code-agent-test-does-not-exist", "server.json");
    assert.throws(() => readServerInfo(missing), AgentUnavailableError);
  });

  test("throws AgentUnavailableError on invalid JSON", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "code-agent-test-"));
    const file = path.join(dir, "server.json");
    fs.writeFileSync(file, "not json");
    assert.throws(() => readServerInfo(file), AgentUnavailableError);
  });

  test("throws AgentUnavailableError when required fields are missing", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "code-agent-test-"));
    const file = path.join(dir, "server.json");
    fs.writeFileSync(file, JSON.stringify({ host: "127.0.0.1" }));
    assert.throws(() => readServerInfo(file), AgentUnavailableError);
  });

  test("parses a valid token file", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "code-agent-test-"));
    const file = path.join(dir, "server.json");
    fs.writeFileSync(file, JSON.stringify({ host: "127.0.0.1", port: 8765, token: "abc123" }));
    const info = readServerInfo(file);
    assert.equal(info.host, "127.0.0.1");
    assert.equal(info.port, 8765);
    assert.equal(info.token, "abc123");
  });
});

let fakeServerTokenCounter = 0;

/** A minimal stand-in for agent/server.py, just enough to exercise
 * AgentClient's request/response and NDJSON-streaming handling. Each
 * instance gets its own distinct token (like a real `code-agent serve`
 * restart would) rather than a shared constant, so tests can tell apart
 * "talking to the same server" from "talking to a different one". */
function startFakeServer(): Promise<{ server: http.Server; port: number; token: string }> {
  fakeServerTokenCounter += 1;
  const token = `test-token-${fakeServerTokenCounter}`;
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const auth = req.headers.authorization;
      if (req.url === "/health") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "ok", backend: "ollama", model: "qwen2.5-coder:3b", ollama_host: "http://localhost:11434", ollama_connected: true }));
        return;
      }
      if (auth !== `Bearer ${token}`) {
        res.writeHead(401, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "unauthorized" }));
        return;
      }
      if (req.url === "/chat" && req.method === "POST") {
        res.writeHead(200, { "Content-Type": "application/x-ndjson", "Transfer-Encoding": "chunked" });
        res.write(JSON.stringify({ type: "content", text: "Hello" }) + "\n");
        res.write(JSON.stringify({ type: "final", text: "Hello" }) + "\n");
        res.end();
        return;
      }
      if (req.url?.startsWith("/task/status")) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ goal: null, plan: null, files_inspected: [] }));
        return;
      }
      if (req.url === "/task/stop" || req.url === "/task/new" || req.url === "/chat/confirm") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
        return;
      }
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "not found" }));
    });
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      resolve({ server, port, token });
    });
  });
}

describe("AgentClient against a fake local server", () => {
  let server: http.Server;
  let tokenFile: string;

  before(async () => {
    const started = await startFakeServer();
    server = started.server;
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "code-agent-test-"));
    tokenFile = path.join(dir, "server.json");
    fs.writeFileSync(tokenFile, JSON.stringify({ host: "127.0.0.1", port: started.port, token: started.token }));
  });

  after(() => {
    server.close();
  });

  test("health() succeeds without requiring the token to be valid", async () => {
    const client = new AgentClient(tokenFile);
    const health = await client.health();
    assert.equal(health.status, "ok");
    assert.equal(health.model, "qwen2.5-coder:3b");
    assert.equal(health.ollama_connected, true);
  });

  test("chat() streams NDJSON events progressively via onEvent", async () => {
    const client = new AgentClient(tokenFile);
    const events: unknown[] = [];
    await client.chat("/some/workspace", "hi", (event) => events.push(event));
    assert.equal(events.length, 2);
    assert.deepEqual(events[0], { type: "content", text: "Hello" });
    assert.deepEqual(events[1], { type: "final", text: "Hello" });
  });

  test("taskStatus() returns parsed JSON", async () => {
    const client = new AgentClient(tokenFile);
    const status = await client.taskStatus("/some/workspace");
    assert.equal(status.goal, null);
  });

  test("taskStop()/taskNew()/chatConfirm() all resolve ok", async () => {
    const client = new AgentClient(tokenFile);
    assert.deepEqual(await client.taskStop("/x"), { ok: true });
    assert.deepEqual(await client.taskNew("/x"), { ok: true });
    assert.deepEqual(await client.chatConfirm("/x", true), { ok: true });
  });

  test("wrong token raises AgentAuthError", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "code-agent-test-"));
    const badTokenFile = path.join(dir, "server.json");
    const address = server.address();
    const port = typeof address === "object" && address ? address.port : 0;
    fs.writeFileSync(badTokenFile, JSON.stringify({ host: "127.0.0.1", port, token: "wrong" }));
    const client = new AgentClient(badTokenFile);
    await assert.rejects(() => client.taskStatus("/x"), AgentAuthError);
  });
});

describe("AgentClient when the agent is not running", () => {
  test("ECONNREFUSED is translated to AgentUnavailableError", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "code-agent-test-"));
    const tokenFile = path.join(dir, "server.json");
    // Port 1 on loopback should reliably refuse connections without a
    // running server -- simulating "code-agent serve" not having started.
    fs.writeFileSync(tokenFile, JSON.stringify({ host: "127.0.0.1", port: 1, token: "x" }));
    const client = new AgentClient(tokenFile);
    await assert.rejects(() => client.taskStatus("/x"), AgentUnavailableError);
  });
});

describe("AgentClient reconnects after the agent server restarts", () => {
  test("a new port/token written to the token file is picked up without recreating the client", async () => {
    // This is the whole mechanism VS Code's "reconnect after server
    // restart" requirement (Phase 8 section 21) relies on: AgentClient
    // must never cache ServerInfo across calls -- each request reads the
    // token file fresh, so restarting `code-agent serve` (new port, new
    // token) is picked up on the very next request with no VS Code window
    // reload needed.
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "code-agent-test-"));
    const tokenFile = path.join(dir, "server.json");

    const first = await startFakeServer();
    fs.writeFileSync(tokenFile, JSON.stringify({ host: "127.0.0.1", port: first.port, token: first.token }));

    const client = new AgentClient(tokenFile);
    const firstHealth = await client.health();
    assert.equal(firstHealth.status, "ok");

    first.server.close();

    const second = await startFakeServer();
    fs.writeFileSync(tokenFile, JSON.stringify({ host: "127.0.0.1", port: second.port, token: second.token }));

    // No new AgentClient constructed -- the same instance, same object,
    // must succeed against the new server on the very next call.
    const secondHealth = await client.health();
    assert.equal(secondHealth.status, "ok");

    second.server.close();
  });

  test("an old cached token is rejected by the new server, proving the token was actually re-read", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "code-agent-test-"));
    const tokenFile = path.join(dir, "server.json");

    const first = await startFakeServer();
    fs.writeFileSync(tokenFile, JSON.stringify({ host: "127.0.0.1", port: first.port, token: first.token }));
    const client = new AgentClient(tokenFile);
    await client.taskStatus("/x");
    first.server.close();

    const second = await startFakeServer();
    // Deliberately write the OLD token with the NEW port -- if AgentClient
    // were caching the token from the first read, this would succeed; it
    // must instead fail, proving every call re-reads the file.
    fs.writeFileSync(tokenFile, JSON.stringify({ host: "127.0.0.1", port: second.port, token: first.token }));

    await assert.rejects(() => client.taskStatus("/x"), AgentAuthError);
    second.server.close();
  });
});
