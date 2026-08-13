// Webview-side UI logic. Plain JS (no build step) run inside the Webview's
// isolated context -- it never talks to the agent server directly (the CSP
// blocks that); it only postMessage()s to the extension host (panel.ts),
// which does all real networking. This file only renders what it's told
// and forwards user actions (send/approve/reject/stop/new/open file/view
// diff) back to the extension.
(function () {
  const vscode = acquireVsCodeApi();

  const statusIndicator = document.getElementById("status-indicator");
  const modelNameEl = document.getElementById("model-name");
  const taskProgressEl = document.getElementById("task-progress");
  const messagesEl = document.getElementById("messages");
  const inputBox = document.getElementById("input-box");
  const sendBtn = document.getElementById("send-btn");
  const stopBtn = document.getElementById("stop-task-btn");
  const newTaskBtn = document.getElementById("new-task-btn");

  const FILE_REF_RE = /([A-Za-z0-9_][A-Za-z0-9_\-./]*\.[A-Za-z]{1,10}):(\d+)/g;

  let currentAgentParagraph = null;
  let pendingApprovalBox = null;

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  /** Renders text with any "path/to/file.ext:123" references turned into
   * clickable spans, without ever using innerHTML on untrusted model
   * output (built entirely from text nodes + our own elements). */
  function appendTextWithFileLinks(container, text) {
    let lastIndex = 0;
    FILE_REF_RE.lastIndex = 0;
    let match;
    while ((match = FILE_REF_RE.exec(text)) !== null) {
      if (match.index > lastIndex) {
        container.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      const link = document.createElement("span");
      link.className = "file-link";
      link.textContent = match[0];
      const filePath = match[1];
      const line = parseInt(match[2], 10);
      link.addEventListener("click", () => {
        vscode.postMessage({ type: "openFile", path: filePath, line });
      });
      container.appendChild(link);
      lastIndex = FILE_REF_RE.lastIndex;
    }
    if (lastIndex < text.length) {
      container.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
  }

  function appendUserMessage(text) {
    pendingApprovalBox = null;
    currentAgentParagraph = null;
    const div = document.createElement("div");
    div.className = "message user";
    const role = document.createElement("span");
    role.className = "role";
    role.textContent = "You: ";
    div.appendChild(role);
    appendTextWithFileLinks(div, text);
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function ensureAgentParagraph() {
    if (currentAgentParagraph) {
      return currentAgentParagraph;
    }
    const div = document.createElement("div");
    div.className = "message agent";
    const role = document.createElement("span");
    role.className = "role";
    role.textContent = "Agent: ";
    div.appendChild(role);
    const textSpan = document.createElement("span");
    textSpan.className = "text";
    div.appendChild(textSpan);
    messagesEl.appendChild(div);
    currentAgentParagraph = { div, textSpan, textSoFar: "" };
    scrollToBottom();
    return currentAgentParagraph;
  }

  function appendAgentContent(text) {
    const p = ensureAgentParagraph();
    p.textSoFar += text;
    p.textSpan.textContent = "";
    appendTextWithFileLinks(p.textSpan, p.textSoFar);
    scrollToBottom();
  }

  function setFinalAgentText(text) {
    const p = ensureAgentParagraph();
    if (p.textSoFar !== text) {
      p.textSoFar = text;
      p.textSpan.textContent = "";
      appendTextWithFileLinks(p.textSpan, text);
    }
    scrollToBottom();
  }

  function appendSubAction(text, cls) {
    const span = document.createElement("span");
    span.className = "sub-action " + cls;
    span.textContent = text;
    messagesEl.appendChild(span);
    scrollToBottom();
  }

  function appendErrorBanner(text) {
    currentAgentParagraph = null;
    const div = document.createElement("div");
    div.className = "error-banner";
    div.textContent = text;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function clearPendingApproval() {
    if (pendingApprovalBox) {
      pendingApprovalBox.querySelectorAll("button").forEach((b) => (b.disabled = true));
      pendingApprovalBox = null;
    }
  }

  function makeApprovalBox() {
    currentAgentParagraph = null;
    const box = document.createElement("div");
    box.className = "approval-box";
    messagesEl.appendChild(box);
    pendingApprovalBox = box;
    return box;
  }

  function addActions(box, approveLabel, rejectLabel, extra) {
    const actions = document.createElement("div");
    actions.className = "actions";
    if (extra) {
      actions.appendChild(extra);
    }
    const approve = document.createElement("button");
    approve.className = "approve";
    approve.textContent = approveLabel;
    approve.addEventListener("click", () => {
      vscode.postMessage({ type: "approve", approved: true });
      clearPendingApproval();
    });
    const reject = document.createElement("button");
    reject.className = "reject";
    reject.textContent = rejectLabel;
    reject.addEventListener("click", () => {
      vscode.postMessage({ type: "approve", approved: false });
      clearPendingApproval();
    });
    actions.appendChild(approve);
    actions.appendChild(reject);
    box.appendChild(actions);
  }

  function renderConfirmChange(change) {
    const box = makeApprovalBox();
    const title = document.createElement("div");
    title.textContent = (change.kind === "create" ? "New file: " : "Modified file: ") + change.path;
    box.appendChild(title);
    const viewDiffBtn = document.createElement("button");
    viewDiffBtn.textContent = "View Diff";
    viewDiffBtn.addEventListener("click", () => {
      vscode.postMessage({
        type: "viewDiff",
        path: change.path,
        oldContent: change.old_content,
        newContent: change.new_content,
        kind: change.kind,
      });
    });
    addActions(box, "Approve", "Reject", viewDiffBtn);
  }

  function renderConfirmCommand(command) {
    const box = makeApprovalBox();
    const title = document.createElement("div");
    title.textContent = "Agent wants to run: " + command.program + " " + (command.args || []).join(" ");
    box.appendChild(title);
    const cwd = document.createElement("div");
    cwd.style.opacity = "0.8";
    cwd.style.fontSize = "0.85em";
    cwd.textContent = "Working directory: " + command.cwd;
    box.appendChild(cwd);
    addActions(box, "Run", "Reject");
  }

  function renderConfirmGitOperation(op) {
    const box = makeApprovalBox();
    const title = document.createElement("div");
    let approveLabel = "Approve";
    if (op.kind === "create_branch") {
      title.textContent = "Create branch: " + op.branch_name;
      approveLabel = "Create Branch";
    } else if (op.kind === "stage") {
      title.textContent = "Stage: " + (op.paths || []).join(", ");
      if (op.sensitive_paths && op.sensitive_paths.length) {
        const warn = document.createElement("div");
        warn.className = "error-banner";
        warn.textContent = "WARNING: includes sensitive file(s): " + op.sensitive_paths.join(", ");
        box.appendChild(title);
        box.appendChild(warn);
        addActions(box, "Stage Anyway", "Cancel");
        return;
      }
      approveLabel = "Stage";
    } else if (op.kind === "commit") {
      title.textContent = "Commit: " + op.message;
      approveLabel = "Commit";
      const files = document.createElement("div");
      files.style.opacity = "0.8";
      files.style.fontSize = "0.85em";
      files.textContent = "Staged files: " + (op.expected_staged_files || []).join(", ");
      box.appendChild(title);
      box.appendChild(files);
      addActions(box, approveLabel, "Cancel");
      return;
    } else if (op.kind === "push") {
      const destination = op.remote_url ? op.remote + " (" + op.remote_url + ")" : op.remote;
      title.textContent = "Push branch '" + op.branch_name + "' to " + destination;
      approveLabel = "Push";
      const preview = document.createElement("pre");
      preview.textContent = op.commits_preview || "";
      box.appendChild(title);
      box.appendChild(preview);
      addActions(box, approveLabel, "Cancel");
      return;
    }
    box.appendChild(title);
    addActions(box, approveLabel, "Cancel");
  }

  function renderConfirmPlan(plan) {
    const box = makeApprovalBox();
    const title = document.createElement("div");
    title.textContent = "Proposed plan: " + plan.goal;
    box.appendChild(title);
    const list = document.createElement("pre");
    list.textContent = (plan.steps || []).map((s) => s.id + ". " + s.description).join("\n");
    box.appendChild(list);
    addActions(box, "Proceed", "Cancel");
  }

  function handleAgentEvent(event) {
    switch (event.type) {
      case "tool_call":
        appendSubAction(event.display, "ok");
        return;
      case "tool_error":
        appendSubAction(event.message, "error");
        return;
      case "repetition_detected":
        appendSubAction(
          "Repeated " + event.name + " call " + event.count + " times without progress -- asked the agent to change approach.",
          "error"
        );
        return;
      case "confirm":
        renderConfirmChange(event.change);
        return;
      case "change_applied":
        appendSubAction("Change applied: " + event.path, "ok");
        return;
      case "change_rejected":
        appendSubAction("Change rejected: " + event.path, "rejected");
        return;
      case "confirm_command":
        renderConfirmCommand(event.command);
        return;
      case "command_started":
        appendSubAction("Running " + event.display + "...", "pending");
        return;
      case "command_result": {
        const r = event.result;
        const label = r.timed_out ? "Command timed out" : "Command finished (exit " + r.exit_code + ")";
        appendSubAction(label, r.timed_out || r.exit_code ? "error" : "ok");
        return;
      }
      case "command_rejected":
        appendSubAction("Command not executed.", "rejected");
        return;
      case "confirm_git_operation":
        renderConfirmGitOperation(event.operation);
        return;
      case "git_operation_applied":
        appendSubAction(event.message, "ok");
        return;
      case "git_operation_rejected":
        appendSubAction("Git operation not performed.", "rejected");
        return;
      case "confirm_plan":
        renderConfirmPlan(event.plan);
        return;
      case "plan_approved":
        appendSubAction("Plan approved.", "ok");
        return;
      case "plan_rejected":
        appendSubAction("Plan not approved.", "rejected");
        return;
      case "content":
        appendAgentContent(event.text);
        return;
      case "final":
        setFinalAgentText(event.text);
        return;
      case "retry":
        appendSubAction(
          "Connection to Ollama failed (" + event.reason + "). Retrying (" + event.attempt + "/" + (event.max_attempts - 1) + ")...",
          "pending"
        );
        return;
      case "error":
        appendErrorBanner("Agent error: " + event.message);
        return;
      case "cancelled":
        appendErrorBanner("Task stopped.");
        return;
      case "max_iterations":
        appendErrorBanner("Stopped after too many steps without a final answer. Try a more specific request.");
        return;
      default:
        return;
    }
  }

  function renderTaskStatus(status) {
    taskProgressEl.textContent = "";
    const hasPlan = status && status.plan;
    const hasFiles =
      status && ((status.files_inspected && status.files_inspected.length) || (status.files_modified && status.files_modified.length));
    if (!hasPlan && !hasFiles) {
      taskProgressEl.classList.remove("visible");
      return;
    }
    taskProgressEl.classList.add("visible");
    if (hasPlan) {
      const header = document.createElement("div");
      header.textContent = "Plan: " + status.plan.goal;
      taskProgressEl.appendChild(header);
      for (const step of status.plan.steps) {
        const line = document.createElement("span");
        line.className = "plan-step " + step.status;
        const marker = { completed: "✓", in_progress: "→", blocked: "✗", failed: "✗", pending: "○" }[step.status] || "○";
        line.textContent = marker + " " + step.id + ". " + step.description + (step.note ? " (" + step.note + ")" : "");
        taskProgressEl.appendChild(line);
      }
    }
    if (status.files_modified && status.files_modified.length) {
      const line = document.createElement("div");
      line.textContent = "Modified: " + status.files_modified.join(", ");
      taskProgressEl.appendChild(line);
    }
  }

  function setStatus(connected, model, ollamaConnected, workspaceOk) {
    if (connected) {
      statusIndicator.className = "online";
      statusIndicator.textContent = "● Connected";
      let suffix = "";
      if (ollamaConnected === false) suffix += " (Ollama not connected)";
      if (workspaceOk === false) suffix += " (workspace folder not found)";
      modelNameEl.textContent = model ? "Model: " + model + suffix : "";
    } else {
      statusIndicator.className = "offline";
      statusIndicator.textContent = "○ Offline";
      modelNameEl.textContent = "Agent server unreachable -- run: code-agent serve";
    }
  }

  function send() {
    const text = inputBox.value.trim();
    if (!text) {
      return;
    }
    vscode.postMessage({ type: "sendMessage", text });
    inputBox.value = "";
  }

  sendBtn.addEventListener("click", send);
  inputBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  stopBtn.addEventListener("click", () => vscode.postMessage({ type: "stopTask" }));
  newTaskBtn.addEventListener("click", () => vscode.postMessage({ type: "newTask" }));

  window.addEventListener("message", (event) => {
    const msg = event.data;
    switch (msg.type) {
      case "status":
        setStatus(msg.connected, msg.model, msg.ollamaConnected, msg.workspaceOk);
        return;
      case "taskStatus":
        renderTaskStatus(msg.status);
        return;
      case "userMessage":
        appendUserMessage(msg.text);
        return;
      case "agentEvent":
        handleAgentEvent(msg.event);
        return;
      case "error":
        appendErrorBanner(msg.message);
        return;
      case "prefill":
        inputBox.value = msg.text;
        inputBox.focus();
        return;
      case "taskReset":
        messagesEl.textContent = "";
        taskProgressEl.textContent = "";
        taskProgressEl.classList.remove("visible");
        currentAgentParagraph = null;
        pendingApprovalBox = null;
        return;
      default:
        return;
    }
  });
})();
