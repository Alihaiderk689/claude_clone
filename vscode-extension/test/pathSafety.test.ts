/**
 * Tests for pathSafety.ts's workspace confinement -- this is what stops a
 * clickable "file:line" reference the model wrote into its reply from
 * opening a file outside the project folder (path traversal via "../"
 * segments, or an absolute path pointing elsewhere). No `vscode` import
 * needed, so this runs with plain node:test.
 */
import * as assert from "node:assert/strict";
import * as path from "node:path";
import { describe, test } from "node:test";

import { resolveWithinWorkspace } from "../src/pathSafety";

const ROOT = path.resolve("/tmp/example-workspace");

describe("resolveWithinWorkspace", () => {
  test("a plain relative path inside the workspace resolves normally", () => {
    const result = resolveWithinWorkspace(ROOT, "backend/auth.py");
    assert.equal(result, path.join(ROOT, "backend", "auth.py"));
  });

  test("the workspace root itself resolves", () => {
    const result = resolveWithinWorkspace(ROOT, ".");
    assert.equal(result, ROOT);
  });

  test("a relative path with a few '../' segments that still lands inside is fine", () => {
    const result = resolveWithinWorkspace(ROOT, "backend/../backend/auth.py");
    assert.equal(result, path.join(ROOT, "backend", "auth.py"));
  });

  test("traversal that escapes the workspace via '../' is rejected", () => {
    const result = resolveWithinWorkspace(ROOT, "../../etc/passwd");
    assert.equal(result, null);
  });

  test("traversal buried in the middle of the reference is rejected", () => {
    const result = resolveWithinWorkspace(ROOT, "src/../../../../etc/passwd.conf");
    assert.equal(result, null);
  });

  test("an absolute path outside the workspace is rejected", () => {
    const result = resolveWithinWorkspace(ROOT, "/etc/passwd");
    assert.equal(result, null);
  });

  test("an absolute path that happens to be inside the workspace is allowed", () => {
    const inside = path.join(ROOT, "backend", "auth.py");
    const result = resolveWithinWorkspace(ROOT, inside);
    assert.equal(result, inside);
  });

  test("a sibling directory that merely shares a name prefix is rejected", () => {
    // e.g. workspace "/tmp/example-workspace" vs "/tmp/example-workspace-evil"
    // -- a naive startsWith(root) check (without the path separator) would
    // wrongly allow this.
    const sibling = ROOT + "-evil/secrets.txt";
    const result = resolveWithinWorkspace(ROOT, sibling);
    assert.equal(result, null);
  });
});
