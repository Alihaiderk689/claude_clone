/**
 * The chat Webview panel: renders agent turns, approval prompts, task
 * progress, and the Connected/Offline + model status indicator. All state
 * (task_state, conversation, plan, approvals) lives on the Python agent;
 * this panel only displays what the server sends and forwards what the
 * user clicks back to it -- see agent/server.py for the actual state.
 */
import * as vscode from "vscode";
import { AgentClient, AgentEvent, AgentMode } from "./client";
import { DiffContentProvider } from "./diffProvider";
import { resolveWithinWorkspace } from "./pathSafety";

const HEALTH_POLL_MS = 5000;
const TASK_REFRESHING_EVENT_TYPES = new Set([
  "plan_approved",
  "plan_rejected",
  "change_applied",
  "change_rejected",
  "command_result",
  "command_rejected",
  "git_operation_applied",
  "git_operation_rejected",
  "final",
  "error",
  "cancelled",
  "max_iterations",
]);

function describeError(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let text = "";
  for (let i = 0; i < 32; i++) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}

export async function openFileAtLine(workspaceRoot: string, relPath: string, line?: number): Promise<void> {
  const resolved = resolveWithinWorkspace(workspaceRoot, relPath);
  if (!resolved) {
    vscode.window.showErrorMessage(
      `Local Coding Assistant: refusing to open "${relPath}" -- it resolves outside the project folder.`
    );
    return;
  }
  const uri = vscode.Uri.file(resolved);
  try {
    const doc = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(doc, { preview: true });
    if (line && line > 0) {
      const pos = new vscode.Position(Math.max(0, line - 1), 0);
      editor.selection = new vscode.Selection(pos, pos);
      editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
    }
  } catch {
    vscode.window.showErrorMessage(`Local Coding Assistant: could not open "${relPath}".`);
  }
}

export class ChatPanel {
  static current: ChatPanel | undefined;

  private readonly panel: vscode.WebviewPanel;
  private readonly disposables: vscode.Disposable[] = [];
  private healthTimer: ReturnType<typeof setInterval> | undefined;
  private turnInFlight = false;

  static createOrShow(
    extensionUri: vscode.Uri,
    client: AgentClient,
    workspaceRoot: string,
    diffProvider: DiffContentProvider
  ): ChatPanel {
    if (ChatPanel.current) {
      if (ChatPanel.current.workspaceRoot !== workspaceRoot) {
        // A different workspace root was requested than the one the open
        // panel is bound to -- never silently mix the two; start fresh
        // rather than risk showing project B's conversation while talking
        // to project A's session.
        ChatPanel.current.dispose();
      } else {
        ChatPanel.current.panel.reveal();
        return ChatPanel.current;
      }
    }
    const panel = vscode.window.createWebviewPanel(
      "codeAgentChat",
      "Local Coding Assistant",
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, "media")],
      }
    );
    ChatPanel.current = new ChatPanel(panel, extensionUri, client, workspaceRoot, diffProvider);
    return ChatPanel.current;
  }

  private constructor(
    panel: vscode.WebviewPanel,
    extensionUri: vscode.Uri,
    private readonly client: AgentClient,
    private readonly workspaceRoot: string,
    private readonly diffProvider: DiffContentProvider
  ) {
    this.panel = panel;
    this.panel.webview.html = this.buildHtml(extensionUri);
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage((msg) => this.handleMessage(msg), null, this.disposables);

    void this.pollHealth();
    this.healthTimer = setInterval(() => void this.pollHealth(), HEALTH_POLL_MS);
    void this.refreshTaskStatus();
  }

  /** Used by explainSelection/fixSelection (auto-send) and
   * askAboutSelection (prefill only, user finishes typing and sends). */
  sendMessage(text: string, autoSend = true): void {
    this.panel.reveal();
    if (autoSend) {
      void this.runChat(text);
    } else {
      this.post({ type: "prefill", text });
    }
  }

  /** Called by the standalone "New Task" command when it's invoked from
   * outside the webview (command palette) rather than the panel's button. */
  notifyTaskReset(): void {
    this.post({ type: "taskReset" });
    void this.refreshTaskStatus();
  }

  private post(message: unknown): void {
    void this.panel.webview.postMessage(message);
  }

  private async pollHealth(): Promise<void> {
    try {
      const health = await this.client.health(this.workspaceRoot);
      this.post({
        type: "status",
        connected: true,
        model: health.model,
        ollamaConnected: health.ollama_connected,
        workspaceOk: health.workspace !== "not_found",
      });
    } catch {
      this.post({ type: "status", connected: false });
    }
  }

  private async refreshTaskStatus(): Promise<void> {
    try {
      const status = await this.client.taskStatus(this.workspaceRoot);
      this.post({ type: "taskStatus", status });
    } catch {
      // Health indicator already communicates an offline agent; nothing
      // further to show here.
    }
  }

  private async handleMessage(msg: { type: string; [key: string]: unknown }): Promise<void> {
    switch (msg.type) {
      case "sendMessage":
        await this.runChat(msg.text as string);
        return;
      case "approve":
        try {
          await this.client.chatConfirm(this.workspaceRoot, Boolean(msg.approved));
        } catch (err) {
          this.post({ type: "error", message: describeError(err) });
        }
        return;
      case "stopTask":
        try {
          await this.client.taskStop(this.workspaceRoot);
        } catch (err) {
          this.post({ type: "error", message: describeError(err) });
        }
        return;
      case "newTask":
        try {
          await this.client.taskNew(this.workspaceRoot);
          this.post({ type: "taskReset" });
          await this.refreshTaskStatus();
        } catch (err) {
          this.post({ type: "error", message: describeError(err) });
        }
        return;
      case "setMode":
        try {
          await this.client.setMode(this.workspaceRoot, msg.mode as AgentMode);
          await this.refreshTaskStatus();
        } catch (err) {
          this.post({ type: "error", message: describeError(err) });
        }
        return;
      case "openFile":
        await openFileAtLine(this.workspaceRoot, msg.path as string, msg.line as number | undefined);
        return;
      case "viewDiff":
        try {
          await this.diffProvider.showDiff(
            msg.path as string,
            (msg.oldContent as string | null) ?? null,
            msg.newContent as string,
            msg.kind as string
          );
        } catch (err) {
          this.post({ type: "error", message: describeError(err) });
        }
        return;
      default:
        return;
    }
  }

  private async runChat(text: string): Promise<void> {
    if (this.turnInFlight) {
      this.post({ type: "error", message: "A turn is already in progress." });
      return;
    }
    this.turnInFlight = true;
    this.post({ type: "userMessage", text });
    try {
      await this.client.chat(this.workspaceRoot, text, (event: AgentEvent) => {
        this.post({ type: "agentEvent", event });
        if (TASK_REFRESHING_EVENT_TYPES.has(event.type as string)) {
          void this.refreshTaskStatus();
        }
      });
    } catch (err) {
      this.post({ type: "error", message: describeError(err) });
    } finally {
      this.turnInFlight = false;
      void this.refreshTaskStatus();
    }
  }

  dispose(): void {
    if (ChatPanel.current === this) {
      ChatPanel.current = undefined;
    }
    if (this.healthTimer) {
      clearInterval(this.healthTimer);
    }
    this.panel.dispose();
    while (this.disposables.length) {
      this.disposables.pop()?.dispose();
    }
  }

  private buildHtml(extensionUri: vscode.Uri): string {
    const webview = this.panel.webview;
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", "main.js"));
    const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", "main.css"));
    const nonce = getNonce();
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; img-src ${webview.cspSource}; script-src 'nonce-${nonce}';" />
<link href="${styleUri}" rel="stylesheet" />
<title>Local Coding Assistant</title>
</head>
<body>
  <div id="status-bar">
    <span id="status-indicator" class="offline">&#9675; Offline</span>
    <span id="model-name"></span>
    <span class="spacer"></span>
    <span class="mode-toggle" role="group" aria-label="Approval mode">
      <button id="mode-manual-btn" class="mode-btn active" title="Ask before every edit">Manual</button>
      <button id="mode-auto-btn" class="mode-btn" title="Auto-approve edits and plans -- commands and Git operations still ask every time">Auto</button>
    </span>
    <button id="stop-task-btn" title="Stop the current task">Stop Task</button>
    <button id="new-task-btn" title="Start a new task">New Task</button>
  </div>
  <div id="task-progress"></div>
  <div id="messages"></div>
  <div id="input-area">
    <textarea id="input-box" placeholder="Ask about this project..." rows="2"></textarea>
    <button id="send-btn">Send</button>
  </div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }
}
