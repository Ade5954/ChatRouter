/* ChatRouter Console — minimal management UI.
 * Reads admin status and routing decisions straight from the gateway API.
 * Keys live only in this browser's localStorage.
 */
"use strict";

const $ = (sel) => document.querySelector(sel);

const store = {
  get admin() { return localStorage.getItem("cr_admin") || ""; },
  set admin(v) { v ? localStorage.setItem("cr_admin", v) : localStorage.removeItem("cr_admin"); },
  get tenant() { return localStorage.getItem("cr_tenant") || ""; },
  set tenant(v) { v ? localStorage.setItem("cr_tenant", v) : localStorage.removeItem("cr_tenant"); },
};

let currentView = "dashboard";
let pollTimer = null;

/* ---------------- connection status ---------------- */

async function ping() {
  const dot = $("#conn-status");
  const text = $("#conn-text");
  try {
    const res = await fetch("/healthz", { cache: "no-store" });
    if (res.ok) {
      dot.className = "conn-dot ok";
      text.textContent = "已连接";
    } else {
      dot.className = "conn-dot warn";
      text.textContent = `HTTP ${res.status}`;
    }
  } catch {
    dot.className = "conn-dot";
    text.textContent = "无法连接";
  }
}

/* ---------------- auth & login ---------------- */

function loadKeysFromUrl() {
  const q = new URLSearchParams(location.search);
  if (q.get("admin")) store.admin = q.get("admin");
  if (q.get("tenant")) store.tenant = q.get("tenant");
}

function renderLogin() {
  const needs = !store.admin && !store.tenant;
  $("#login").classList.toggle("hidden", !needs);
  $("#admin-key").value = store.admin;
  $("#tenant-key").value = store.tenant;
}

function apiError(res, body) {
  let msg = `HTTP ${res.status}`;
  if (body && body.error && body.error.message) msg = body.error.message;
  else if (body && body.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  return msg;
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, { cache: "no-store", ...options });
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error(apiError(res, body));
  return body;
}

function adminHeaders() {
  const h = {};
  if (store.admin) h["x-admin-key"] = store.admin;
  return h;
}
function tenantHeaders() {
  const h = {};
  if (store.tenant) h["Authorization"] = `Bearer ${store.tenant}`;
  return h;
}

/* ---------------- helpers ---------------- */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function tierBadge(tier) {
  const t = tier || "";
  return `<span class="badge ${esc(t)}">${esc(t)}</span>`;
}

function barClass(ratio) {
  if (ratio >= 0.9) return "bad";
  if (ratio >= 0.65) return "warn";
  return "good";
}

function meter(name, ratio, display, cls) {
  const pct = Math.max(0, Math.min(100, ratio * 100));
  const fill = cls || barClass(ratio);
  return `
    <div class="meter">
      <div class="label"><span>${esc(name)}</span><span>${esc(display ?? (pct.toFixed(0) + "%"))}</span></div>
      <div class="bar"><div class="fill ${fill}" style="width:${pct}%"></div></div>
    </div>`;
}

/* ---------------- Dashboard ---------------- */

async function loadDashboard() {
  const grid = $("#model-grid");
  const tenants = $("#tenant-list");
  if (!store.admin) {
    grid.innerHTML = '<div class="empty">需要配置 admin key 才能查看。</div>';
    tenants.innerHTML = "";
    return;
  }
  try {
    const data = await fetchJSON("/admin/status", { headers: adminHeaders() });
    $("#last-updated").textContent = new Date().toLocaleTimeString();

    grid.innerHTML = data.models.map((m) => {
      const circuit = m.circuit ? m.circuit.state : "closed";
      const load = m.load || {};
      const stats = m.stats || {};
      const quality = m.effective_quality ?? m.quality_prior;
      const saturation = load.utilisation ?? 0;
      const latency = stats.latency_ema_ms ?? m.latency_prior_ms;
      const err = stats.success_rate !== undefined ? (1 - stats.success_rate) * 100 : null;
      const circuitBadge =
        circuit === "open" ? '<span class="badge circuit-open">熔断</span>'
        : circuit === "half_open" ? '<span class="badge circuit-half_open">半开</span>'
        : '<span class="badge circuit-closed">正常</span>';
      return `
      <div class="card">
        <div class="name">${esc(m.id)} ${tierBadge(m.tier)} ${circuitBadge}</div>
        <div class="provider">${esc(m.provider)} · ${esc(m.tier)} 档</div>
        ${meter("综合质量", quality, quality.toFixed(3))}
        ${meter("负载", saturation)}
        ${latency != null ? `<div class="meter"><div class="label"><span>延迟 EMA</span><span>${Math.round(latency)}ms</span></div></div>` : ""}
        ${err != null ? `<div class="meter"><div class="label"><span>失败率</span><span>${err.toFixed(1)}%</span></div></div>` : ""}
      </div>`;
    }).join("");

    if (!data.models.length) grid.innerHTML = '<div class="empty">没有可用模型。</div>';

    tenants.innerHTML = data.tenants.map((t) => {
      const rl = t.rate_limit || {};
      const q = t.quota || {};
      const rlRatio = rl.rpm_limit ? rl.rpm_used / rl.rpm_limit : null;
      const tpmRatio = rl.tpm_limit ? rl.tpm_used / rl.tpm_limit : null;
      const conRatio = rl.concurrency_limit ? rl.inflight / rl.concurrency_limit : null;
      const reqRatio = q.max_requests ? q.requests / q.max_requests : null;
      const costRatio = q.max_cost_usd ? q.cost_usd / q.max_cost_usd : null;
      const tokRatio = q.max_tokens ? q.tokens / q.max_tokens : null;
      return `
      <div class="card">
        <div class="name">${esc(t.tenant)}</div>
        ${rl.limit_requests != null ? meter("RPM", rlRatio, `${rl.rpm_used}/${rl.rpm_limit}`) : ""}
        ${rl.limit_tokens != null ? meter("TPM", tpmRatio, `${rl.tpm_used}/${rl.tpm_limit}`) : ""}
        ${conRatio != null ? meter("并发", conRatio, `${rl.inflight}/${rl.concurrency_limit}`) : ""}
        ${reqRatio != null ? meter("配额·请求", reqRatio, `${q.requests}/${q.max_requests}`) : ""}
        ${tokRatio != null ? meter("配额·Token", tokRatio, `${q.tokens}/${q.max_tokens}`) : ""}
        ${costRatio != null ? meter("配额·费用", costRatio, `$${q.cost_usd.toFixed(4)}/$${q.max_cost_usd}`) : ""}
      </div>`;
    }).join("");

    if (!data.tenants.length) tenants.innerHTML = '<div class="empty">没有租户数据。</div>';
  } catch (e) {
    grid.innerHTML = `<div class="error-box">Dashboard 加载失败：${esc(e.message)}</div>`;
    tenants.innerHTML = "";
  }
}

function startPoll() {
  stopPoll();
  pollTimer = setInterval(() => {
    if ($("#auto-refresh").checked) loadDashboard();
  }, 5000);
}
function stopPoll() { if (pollTimer) clearInterval(pollTimer); pollTimer = null; }

/* ---------------- Playground ---------------- */

function parseConversation(text) {
  const roles = new Set(["user", "assistant", "system", "tool"]);
  const messages = [];
  text.split("\n").forEach((raw) => {
    const line = raw.trim();
    if (!line) return;
    const m = line.match(/^(\w+)\s*:\s*(.*)$/s);
    if (m && roles.has(m[1].toLowerCase()) && m[2].trim()) {
      messages.push({ role: m[1].toLowerCase(), content: m[2].trim() });
    } else {
      messages.push({ role: "user", content: line });
    }
  });
  return messages;
}

async function runExplain() {
  const status = $("#pg-status");
  const box = $("#decision");
  status.textContent = "分析中…";
  box.classList.add("hidden");
  if (!store.tenant) {
    status.textContent = "需要配置租户 API key。";
    return;
  }
  const messages = parseConversation($("#convo").value);
  if (!messages.length) {
    status.textContent = "请至少输入一条消息。";
    return;
  }
  try {
    const data = await fetchJSON("/v1/routing/explain", {
      method: "POST",
      headers: { ...tenantHeaders(), "content-type": "application/json" },
      body: JSON.stringify({ model: $("#pg-model").value, messages }),
    });
    status.textContent = "";
    renderDecision(data);
  } catch (e) {
    status.textContent = "";
    box.classList.remove("hidden");
    box.innerHTML = `<div class="error-box">分析失败：${esc(e.message)}</div>`;
  }
}

function renderDecision(data) {
  const box = $("#decision");
  const d = data.decision;
  const a = d.assessment;
  box.classList.remove("hidden");

  const signals = a ? Object.entries(a.signals) : [];
  const signalHtml = signals.length
    ? `<div class="panel"><h2>复杂度信号</h2>
        ${signals.filter(([, v]) => v > 0).map(([k, v]) => `
          <div class="signal-row"><span class="name">${esc(k)}</span>
            <div class="bar"><div class="fill" style="width:${(v * 100).toFixed(0)}%"></div></div>
            <span class="val">${v.toFixed(2)}</span></div>`).join("")}
        ${signals.some(([, v]) => v > 0) ? "" : '<div class="muted">所有信号均为 0（极简请求）</div>'}
      </div>` : "";

  const candidates = (d.candidates || []).map((c, i) => {
    const width = (Math.max(0, c.utility) * 100).toFixed(1);
    const isWin = i === 0;
    return `
    <div class="candidate ${isWin ? "winner" : ""}">
      <span class="name">${isWin ? "✓ " : ""}${esc(c.model)} ${tierBadge(c.tier)}</span>
      <div class="bar"><div class="fill" style="width:${width}%"></div></div>
      <span class="meta">效用 ${c.utility.toFixed(3)} · 质量 ${c.quality.toFixed(3)} · 成本 ${c.cost_score.toFixed(2)} · 延迟 ${c.latency_score.toFixed(2)}</span>
    </div>`;
  }).join("");

  const notes = (d.notes || []).length
    ? `<div class="panel"><h2>决策说明</h2><ul class="notes">${d.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>`
    : "";

  box.innerHTML = `
    <div class="panel">
      <div class="decision-header">
        <div>
          <div class="score-big" style="color:${a ? "var(--accent)" : "var(--muted)"}">${a ? a.score.toFixed(3) : "—"}</div>
          <div class="muted">复杂度分数</div>
        </div>
        <div>
          <div class="muted">选中模型</div>
          <div style="font-size:20px;font-weight:600">${esc(d.model)} ${tierBadge(a ? a.tier : "")}</div>
          <div class="muted">理由：${esc(d.reason)}${d.exploration ? "（探索）" : ""}</div>
        </div>
        <div>
          <div class="muted">预计输入</div>
          <div style="font-size:20px;font-weight:600">${a ? a.prompt_tokens_estimate : "—"} tokens</div>
          <div class="muted">轮次：${a ? a.turn_count : "—"} · 降级链：${(d.fallback_chain || []).join(", ") || "—"}</div>
        </div>
      </div>
    </div>
    ${signalHtml}
    <div class="panel"><h2>候选模型效用</h2>${candidates || '<div class="muted">无候选数据</div>'}</div>
    ${notes}`;
}

/* ---------------- nav & wiring ---------------- */

function switchView(view) {
  currentView = view;
  document.querySelectorAll(".nav-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view));
  $("#view-dashboard").classList.toggle("hidden", view !== "dashboard");
  $("#view-playground").classList.toggle("hidden", view !== "playground");
  if (view === "dashboard") loadDashboard();
}

async function loadModelsIntoSelect() {
  const sel = $("#pg-model");
  if (!store.admin) return;
  try {
    const data = await fetchJSON("/admin/status", { headers: adminHeaders() });
    const known = new Set(data.models.map((m) => m.id));
    sel.innerHTML = '<option value="auto">auto</option>' +
      data.models.map((m) => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join("");
    if (!$("#pg-model").value && known.has("auto")) $("#pg-model").value = "auto";
  } catch { /* non-fatal */ }
}

function init() {
  loadKeysFromUrl();

  $("#save-keys").addEventListener("click", () => {
    store.admin = $("#admin-key").value.trim();
    store.tenant = $("#tenant-key").value.trim();
    const ok = $("#key-saved");
    ok.classList.remove("hidden");
    setTimeout(() => ok.classList.add("hidden"), 1500);
    renderLogin();
    switchView(currentView);
    loadModelsIntoSelect();
  });
  $("#clear-keys").addEventListener("click", () => {
    store.admin = ""; store.tenant = "";
    renderLogin();
  });
  $("#admin-key").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#save-keys").click(); });
  $("#tenant-key").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#save-keys").click(); });

  document.querySelectorAll(".nav-btn").forEach((b) =>
    b.addEventListener("click", () => switchView(b.dataset.view)));
  $("#refresh").addEventListener("click", loadDashboard);
  $("#explain-btn").addEventListener("click", runExplain);
  $("#convo").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runExplain();
  });

  renderLogin();
  startPoll();
  ping();
  setInterval(ping, 15000);
  switchView("dashboard");
  loadModelsIntoSelect();
}

document.addEventListener("DOMContentLoaded", init);
