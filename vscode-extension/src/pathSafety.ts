/**
 * Path confinement for file references the agent's (untrusted, model-
 * generated) text produces -- e.g. a clickable "backend/auth.py:42"
 * reference in a chat reply. Mirrors the confinement
 * agent/project.py's ProjectRoot.resolve() enforces on the Python side
 * for every file operation: a relative reference containing "../"
 * segments, or an absolute path, must never resolve outside the
 * workspace root just because the model wrote it into its reply.
 *
 * Deliberately has no `vscode` import (like client.ts) so it's testable
 * with plain node:test, without a VS Code host.
 */
import * as path from "node:path";

/** Returns the resolved absolute path if it stays inside `workspaceRoot`,
 * or null if it would escape (via "../" traversal or an absolute path
 * pointing elsewhere) -- never throws. */
export function resolveWithinWorkspace(workspaceRoot: string, relPath: string): string | null {
  const resolvedRoot = path.resolve(workspaceRoot);
  const candidate = path.isAbsolute(relPath) ? relPath : path.join(resolvedRoot, relPath);
  const resolved = path.resolve(candidate);
  if (resolved !== resolvedRoot && !resolved.startsWith(resolvedRoot + path.sep)) {
    return null;
  }
  return resolved;
}
