/**
 * Command handlers registered in extension.ts. Each one resolves a
 * workspace root defensively (never guessing an unrelated folder) before
 * doing anything, and every actual action goes through AgentClient to the
 * Python agent -- no tool logic, editing, or approval decisions live here.
 */
import * as path from "node:path";
import * as vscode from "vscode";
import { AgentClient } from "./client";
import { ChatPanel } from "./panel";
import { DiffContentProvider } from "./diffProvider";

function describeError(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Picks the workspace root a command should act on. Never falls back to
 * an arbitrary/unrelated directory:
 *   - No folder open at all -> ask the user to open one, return undefined.
 *   - `preferActiveEditor` and an active editor belongs to one of the open
 *     folders -> use that folder (correct choice in a multi-root workspace
 *     for selection-based commands).
 *   - Otherwise -> the first workspace folder, with a warning if there is
 *     more than one so the user knows why (multi-root workspaces beyond
 *     "pick the first folder" are out of scope for this phase).
 */
export function resolveWorkspaceRoot(preferActiveEditor = false): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    vscode.window.showErrorMessage("Local Coding Assistant: open a folder or workspace first.");
    return undefined;
  }

  if (preferActiveEditor) {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
      if (folder) {
        return folder.uri.fsPath;
      }
    }
  }

  if (folders.length > 1) {
    vscode.window.showWarningMessage(
      "Local Coding Assistant: multiple workspace folders are open; using the first one."
    );
  }
  return folders[0].uri.fsPath;
}

interface SelectionContext {
  text: string;
  relPath: string;
  startLine: number;
  endLine: number;
  root: string;
}

function getSelectionContext(): SelectionContext | undefined {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showErrorMessage("Local Coding Assistant: select some code first.");
    return undefined;
  }
  const root = resolveWorkspaceRoot(true);
  if (!root) {
    return undefined;
  }
  const text = editor.document.getText(editor.selection);
  const relPath = path.relative(root, editor.document.uri.fsPath) || path.basename(editor.document.uri.fsPath);
  return {
    text,
    relPath,
    startLine: editor.selection.start.line + 1,
    endLine: editor.selection.end.line + 1,
    root,
  };
}

export function openChat(extensionUri: vscode.Uri, client: AgentClient, diffProvider: DiffContentProvider): void {
  const root = resolveWorkspaceRoot();
  if (!root) {
    return;
  }
  ChatPanel.createOrShow(extensionUri, client, root, diffProvider);
}

function openChatWithSelection(
  extensionUri: vscode.Uri,
  client: AgentClient,
  diffProvider: DiffContentProvider,
  buildPrompt: (ctx: SelectionContext) => string,
  autoSend: boolean
): void {
  const ctx = getSelectionContext();
  if (!ctx) {
    return;
  }
  const panel = ChatPanel.createOrShow(extensionUri, client, ctx.root, diffProvider);
  panel.sendMessage(buildPrompt(ctx), autoSend);
}

export function explainSelection(extensionUri: vscode.Uri, client: AgentClient, diffProvider: DiffContentProvider): void {
  openChatWithSelection(
    extensionUri,
    client,
    diffProvider,
    (ctx) =>
      `Explain this code from ${ctx.relPath}:${ctx.startLine}-${ctx.endLine}:\n\n\`\`\`\n${ctx.text}\n\`\`\``,
    true
  );
}

export function fixSelection(extensionUri: vscode.Uri, client: AgentClient, diffProvider: DiffContentProvider): void {
  openChatWithSelection(
    extensionUri,
    client,
    diffProvider,
    (ctx) =>
      `Fix this code from ${ctx.relPath}:${ctx.startLine}-${ctx.endLine}. Inspect the surrounding file first if ` +
      `you need more context -- don't assume the fix is confined to only these lines:\n\n\`\`\`\n${ctx.text}\n\`\`\``,
    true
  );
}

export function askAboutSelection(extensionUri: vscode.Uri, client: AgentClient, diffProvider: DiffContentProvider): void {
  openChatWithSelection(
    extensionUri,
    client,
    diffProvider,
    (ctx) => `About this code from ${ctx.relPath}:${ctx.startLine}-${ctx.endLine}:\n\n\`\`\`\n${ctx.text}\n\`\`\`\n\n`,
    false
  );
}

export async function stopTask(client: AgentClient): Promise<void> {
  const root = resolveWorkspaceRoot();
  if (!root) {
    return;
  }
  try {
    await client.taskStop(root);
  } catch (err) {
    vscode.window.showErrorMessage(`Local Coding Assistant: ${describeError(err)}`);
  }
}

export async function newTask(client: AgentClient): Promise<void> {
  const root = resolveWorkspaceRoot();
  if (!root) {
    return;
  }
  try {
    await client.taskNew(root);
    ChatPanel.current?.notifyTaskReset();
  } catch (err) {
    vscode.window.showErrorMessage(`Local Coding Assistant: ${describeError(err)}`);
  }
}
