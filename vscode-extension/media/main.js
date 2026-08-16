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
  const modeManualBtn = document.getElementById("mode-manual-btn");
  const modeAutoBtn = document.getElementById("mode-auto-btn");

  const FILE_REF_RE = /([A-Za-z0-9_][A-Za-z0-9_\-./]*\.[A-Za-z]{1,10}):(\d+)/g;

  let currentAgentParagraph = null;
  let pendingApprovalBox = null;

  // Vertical activity timeline (one segment per user turn) -- see
  // ensureTimeline()/resolveStep()/setActiveStep() below. `activeTimelineItem`
  // is the trailing "currently working" node (e.g. "Thinking..."); resolving
  // it converts it in place into a completed step instead of appending a
  // second element, so there's always at most one active node per turn.
  let currentTimeline = null;
  let activeTimelineItem = null;
  // Cached data from confirm/confirm_command events, keyed so the LATER
  // change_applied/command_result event (which the backend sends with no
  // content of its own) can still build a rich preview. Safe because
  // approvals are strictly serial within a turn -- no protocol change on
  // the Python side was needed for this.
  const pendingChangesByPath = new Map();
  let pendingCommand = null;

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
    resetTimelineForNewTurn();
    const div = document.createElement("div");
    div.className = "message user";
    const role = document.createElement("span");
    role.className = "role";
    role.textContent = "You: ";
    div.appendChild(role);
    appendTextWithFileLinks(div, text);
    messagesEl.appendChild(div);
    setActiveStep("Thinking...");
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
    clearActiveStep();
    const p = ensureAgentParagraph();
    if (p.textSoFar !== text) {
      p.textSoFar = text;
      p.textSpan.textContent = "";
      appendTextWithFileLinks(p.textSpan, text);
    }
    scrollToBottom();
  }

  function ensureTimeline() {
    if (currentTimeline) {
      return currentTimeline;
    }
    const div = document.createElement("div");
    div.className = "timeline";
    messagesEl.appendChild(div);
    currentTimeline = div;
    return currentTimeline;
  }

  function resetTimelineForNewTurn() {
    currentTimeline = null;
    activeTimelineItem = null;
    pendingChangesByPath.clear();
    pendingCommand = null;
  }

  function makeDot() {
    const dot = document.createElement("span");
    dot.className = "timeline-dot";
    return dot;
  }

  function buildSimpleLabel(container, text) {
    const label = document.createElement("span");
    label.className = "timeline-label";
    appendTextWithFileLinks(label, text);
    container.appendChild(label);
  }

  function buildRichLabel(container, verb, rest) {
    const label = document.createElement("span");
    label.className = "timeline-label";
    const b = document.createElement("b");
    b.textContent = verb;
    label.appendChild(b);
    if (rest) {
      label.appendChild(document.createTextNode(" " + rest));
    }
    container.appendChild(label);
  }

  /** Converts the current active ("...ing") node in place into a resolved
   * step, or appends a fresh resolved step if there's no active node to
   * reuse -- either way there's exactly one element per logical step, never
   * a disconnected pair. */
  function resolveStep(cls, buildContent) {
    const timeline = ensureTimeline();
    let item = activeTimelineItem;
    if (!item) {
      item = document.createElement("div");
      item.appendChild(makeDot());
      const content = document.createElement("div");
      content.className = "timeline-content";
      item.appendChild(content);
      timeline.appendChild(item);
    }
    item.className = "timeline-item " + cls;
    const content = item.querySelector(".timeline-content");
    content.textContent = "";
    buildContent(content);
    activeTimelineItem = null;
    scrollToBottom();
    return item;
  }

  function appendTimelineStep(text, cls) {
    resolveStep(cls, (content) => buildSimpleLabel(content, text));
  }

  function buildChangeContent(content, change) {
    const verb = change.kind === "create" ? "Write" : "Edit";
    buildRichLabel(content, verb, change.path);
    const newContent = change.new_content || "";
    const lines = newContent.length ? newContent.split("\n") : [];
    const lineCount = lines.length && lines[lines.length - 1] === "" ? lines.length - 1 : lines.length;
    const meta = document.createElement("div");
    meta.className = "timeline-meta";
    meta.textContent = lineCount + (lineCount === 1 ? " line" : " lines");
    content.appendChild(meta);
    if (newContent) {
      const preview = document.createElement("pre");
      preview.className = "timeline-preview";
      const shown = lines.slice(0, 10);
      preview.textContent = shown.join("\n") + (lines.length > 10 ? "\n..." : "");
      content.appendChild(preview);
    }
  }

  function buildCommandContent(content, cmd, resultLabel) {
    buildRichLabel(content, "Bash", null);
    if (cmd) {
      const meta = document.createElement("div");
      meta.className = "timeline-meta";
      meta.textContent = cmd.program + " " + (cmd.args || []).join(" ");
      content.appendChild(meta);
    }
    const box = document.createElement("pre");
    box.className = "timeline-preview";
    box.textContent = "IN " + (cmd ? cmd.cwd : "unknown") + (resultLabel ? "\n" + resultLabel : "");
    content.appendChild(box);
  }

  /** Creates the trailing "currently working" node if none exists, or just
   * updates its text in place if one already does (e.g. a retry message
   * replacing "Thinking..." without adding a second dot). */
  function setActiveStep(text) {
    const timeline = ensureTimeline();
    if (activeTimelineItem) {
      activeTimelineItem.querySelector(".timeline-label").textContent = text;
      scrollToBottom();
      return activeTimelineItem;
    }
    const item = document.createElement("div");
    item.className = "timeline-item active";
    item.appendChild(makeDot());
    const content = document.createElement("div");
    content.className = "timeline-content";
    buildSimpleLabel(content, text);
    item.appendChild(content);
    timeline.appendChild(item);
    activeTimelineItem = item;
    scrollToBottom();
    return item;
  }

  function clearActiveStep() {
    if (activeTimelineItem) {
      activeTimelineItem.remove();
      activeTimelineItem = null;
    }
  }

  function appendErrorBanner(text) {
    currentAgentParagraph = null;
    clearActiveStep();
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

  function renderConfirmChange(change, autoApproved) {
    if (autoApproved) {
      pendingChangesByPath.delete(change.path);
      resolveStep("ok", (content) => buildChangeContent(content, change));
      setActiveStep("Thinking...");
      return;
    }
    setActiveStep("Waiting for your approval...");
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
    setActiveStep("Waiting for your approval...");
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
    setActiveStep("Waiting for your approval...");
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

  function renderConfirmPlan(plan, autoApproved) {
    if (autoApproved) {
      appendTimelineStep("✓ Auto-approved plan: " + plan.goal, "ok");
      setActiveStep("Thinking...");
      return;
    }
    setActiveStep("Waiting for your approval...");
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
        appendTimelineStep(event.display, "ok");
        setActiveStep("Thinking...");
        return;
      case "tool_error":
        appendTimelineStep(event.message, "error");
        setActiveStep("Thinking...");
        return;
      case "repetition_detected":
        appendTimelineStep(
          "Repeated " + event.name + " call " + event.count + " times without progress -- asked the agent to change approach.",
          "error"
        );
        setActiveStep("Thinking...");
        return;
      case "confirm":
        pendingChangesByPath.set(event.change.path, event.change);
        renderConfirmChange(event.change, event.auto_approved);
        return;
      case "change_applied": {
        const change = pendingChangesByPath.get(event.path);
        pendingChangesByPath.delete(event.path);
        resolveStep("ok", (content) => {
          if (change) {
            buildChangeContent(content, change);
          } else {
            buildSimpleLabel(content, "Change applied: " + event.path);
          }
        });
        setActiveStep("Thinking...");
        return;
      }
      case "change_rejected":
        pendingChangesByPath.delete(event.path);
        resolveStep("rejected", (content) => buildSimpleLabel(content, "Change rejected: " + event.path));
        setActiveStep("Thinking...");
        return;
      case "confirm_command":
        pendingCommand = event.command;
        renderConfirmCommand(event.command);
        return;
      case "command_started":
        setActiveStep("Running " + event.display + "...");
        return;
      case "command_result": {
        const r = event.result;
        const ok = !(r.timed_out || r.exit_code);
        const resultLabel = r.timed_out ? "Command timed out" : "Exit code " + r.exit_code;
        const cmd = pendingCommand;
        pendingCommand = null;
        resolveStep(ok ? "ok" : "error", (content) => buildCommandContent(content, cmd, resultLabel));
        setActiveStep("Thinking...");
        return;
      }
      case "command_rejected":
        pendingCommand = null;
        resolveStep("rejected", (content) => buildSimpleLabel(content, "Command not executed."));
        setActiveStep("Thinking...");
        return;
      case "confirm_git_operation":
        renderConfirmGitOperation(event.operation);
        return;
      case "git_operation_applied":
        appendTimelineStep(event.message, "ok");
        setActiveStep("Thinking...");
        return;
      case "git_operation_rejected":
        appendTimelineStep("Git operation not performed.", "rejected");
        setActiveStep("Thinking...");
        return;
      case "confirm_plan":
        renderConfirmPlan(event.plan, event.auto_approved);
        return;
      case "plan_approved":
        appendTimelineStep("Plan approved.", "ok");
        setActiveStep("Thinking...");
        return;
      case "plan_rejected":
        appendTimelineStep("Plan not approved.", "rejected");
        setActiveStep("Thinking...");
        return;
      case "content":
        appendAgentContent(event.text);
        return;
      case "final":
        setFinalAgentText(event.text);
        return;
      case "retry":
        setActiveStep(
          "Connection to Ollama failed (" + event.reason + "). Retrying (" + event.attempt + "/" + (event.max_attempts - 1) + ")..."
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

  function setMode(mode) {
    modeManualBtn.classList.toggle("active", mode !== "auto");
    modeAutoBtn.classList.toggle("active", mode === "auto");
  }

  function renderTaskStatus(status) {
    if (status && status.mode) {
      setMode(status.mode);
    }
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
  modeManualBtn.addEventListener("click", () => {
    setMode("manual");
    vscode.postMessage({ type: "setMode", mode: "manual" });
  });
  modeAutoBtn.addEventListener("click", () => {
    setMode("auto");
    vscode.postMessage({ type: "setMode", mode: "auto" });
  });

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
        resetTimelineForNewTurn();
        return;
      default:
        return;
    }
  });
})();
