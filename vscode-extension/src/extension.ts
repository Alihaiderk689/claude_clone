/**
 * Extension entry point. Registers commands and the diff content provider;
 * everything else lives in client.ts (HTTP/NDJSON transport), panel.ts
 * (webview), diffProvider.ts (native diff integration), and commands.ts
 * (command handler logic). Never spawns the Python agent itself -- the
 * user starts `code-agent serve` separately, and this extension only talks
 * to it over the local HTTP server.
 */
import * as vscode from "vscode";
import { AgentClient } from "./client";
import * as commands from "./commands";
import { DiffContentProvider, DIFF_SCHEME } from "./diffProvider";

export function activate(context: vscode.ExtensionContext): void {
  const client = new AgentClient();
  const diffProvider = new DiffContentProvider();

  context.subscriptions.push(vscode.workspace.registerTextDocumentContentProvider(DIFF_SCHEME, diffProvider));

  context.subscriptions.push(
    vscode.commands.registerCommand("codeAgent.openChat", () =>
      commands.openChat(context.extensionUri, client, diffProvider)
    ),
    vscode.commands.registerCommand("codeAgent.explainSelection", () =>
      commands.explainSelection(context.extensionUri, client, diffProvider)
    ),
    vscode.commands.registerCommand("codeAgent.fixSelection", () =>
      commands.fixSelection(context.extensionUri, client, diffProvider)
    ),
    vscode.commands.registerCommand("codeAgent.askAboutSelection", () =>
      commands.askAboutSelection(context.extensionUri, client, diffProvider)
    ),
    vscode.commands.registerCommand("codeAgent.stopTask", () => commands.stopTask(client)),
    vscode.commands.registerCommand("codeAgent.newTask", () => commands.newTask(client))
  );
}

export function deactivate(): void {
  // Nothing to tear down here -- the agent server is a separate process
  // (started via `code-agent serve`) and outlives any single VS Code
  // window on purpose, so other windows/workspaces can keep using it.
}
