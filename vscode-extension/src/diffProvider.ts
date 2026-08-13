/**
 * Renders a proposed file change in VS Code's native diff editor.
 *
 * This never writes to disk and never decides anything about whether a
 * change is applied -- it only visualizes the same old/new content that
 * already arrived in a "confirm" event from the Python agent. Approving or
 * rejecting still goes through ChatPanel -> AgentClient.chatConfirm() ->
 * the Python agent, which is the only thing that ever calls apply_change().
 */
import * as vscode from "vscode";

export const DIFF_SCHEME = "code-agent-diff";

export class DiffContentProvider implements vscode.TextDocumentContentProvider {
  private readonly contents = new Map<string, string>();
  private readonly emitter = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.emitter.event;

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.toString()) ?? "";
  }

  async showDiff(relPath: string, oldContent: string | null, newContent: string, kind: string): Promise<void> {
    const stamp = Date.now();
    const leftUri = vscode.Uri.parse(`${DIFF_SCHEME}:${relPath}?before#${stamp}`);
    const rightUri = vscode.Uri.parse(`${DIFF_SCHEME}:${relPath}?after#${stamp}`);
    this.contents.set(leftUri.toString(), kind === "create" ? "" : oldContent ?? "");
    this.contents.set(rightUri.toString(), newContent);
    const title = kind === "create" ? `${relPath} (new file)` : `${relPath} (proposed change)`;
    await vscode.commands.executeCommand("vscode.diff", leftUri, rightUri, title);
  }
}
