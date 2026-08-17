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
  box.classList.remove("hidden");
  box.innerHTML = decisionHtml(data);
}

/* Decision breakdown as pure HTML; shared by the Playground and the live
 * chat sidebar so the same explanation appears in both places. */
function decisionHtml(data) {
  const d = data.decision;
  const a = d.assessment;
  return `
    <div class="panel">
      <div class="decision-header">
        <div>
          <div class="score-big" style="color:${a ? "var(--accent)" : "var(--muted)"}">${a ? a.score.toFixed(3) : "—"}</div>
          <div class="muted">复杂度分数</div>
        </div>
        <div>
          <div class="muted">ChatRouter 选中</div>
          <div class="winner-name">${esc(d.model)} ${tierBadge(a ? a.tier : "")}</div>
          <div class="muted">理由：${esc(d.reason)}${d.exploration ? "（探索）" : ""}</div>
        </div>
        <div>
          <div class="muted">预计输入</div>
          <div class="winner-name">${a ? a.prompt_tokens_estimate : "—"} tokens</div>
          <div class="muted">轮次：${a ? a.turn_count : "—"} · 降级链：${(d.fallback_chain || []).join(", ") || "—"}</div>
        </div>
      </div>
    </div>
    ${renderCandidates(d)}
    ${renderSignals(a)}
    ${(d.notes || []).length ? `<div class="panel"><h2>决策说明</h2><ul class="notes">${d.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>` : ""}`;
}

/* Candidate ranking with the utility decomposed into its drivers. */
function renderCandidates(d) {
  const cands = d.candidates || [];
  if (!cands.length) return "";
  const maxUtil = Math.max(...cands.map((c) => Math.max(0, c.utility)), 1e-6);
  const items = cands.map((c, i) => {
    const width = ((Math.max(0, c.utility) / maxUtil) * 100).toFixed(1);
    const isWin = i === 0;
    const dims = [
      ["质量", c.quality, "good"],
      ["成本", c.cost_score, ""],
      ["延迟", c.latency_score, ""],
      ["负载", c.load_score, ""],
    ].map(([n, v, cls]) => `
      <div class="dim">
        <span class="dim-name">${n}</span>
        <div class="bar"><div class="fill ${cls}" style="width:${(v * 100).toFixed(0)}%"></div></div>
      </div>`).join("");
    return `
    <div class="candidate ${isWin ? "winner" : ""}">
      <div class="cand-head">
        <span class="name">${isWin ? "✓ " : ""}${esc(c.model)} ${tierBadge(c.tier)}</span>
        <span class="util">效用 ${c.utility.toFixed(3)}</span>
      </div>
      <div class="bar"><div class="fill" style="width:${width}%"></div></div>
      <div class="dims">${dims}</div>
      <div class="cand-meta">档距惩罚 ${c.tier_penalty.toFixed(3)} · 探索加成 ${c.exploration_bonus.toFixed(3)}</div>
    </div>`;
  }).join("");
  return `<div class="panel"><h2>候选模型效用分解</h2>${items}</div>`;
}

function renderSignals(a) {
  if (!a) return "";
  const signals = Object.entries(a.signals);
  const active = signals.filter(([, v]) => v > 0);
  return `<div class="panel"><h2>复杂度信号</h2>
    ${active.length
      ? active.map(([k, v]) => `
          <div class="signal-row"><span class="name">${esc(k)}</span>
            <div class="bar"><div class="fill" style="width:${(v * 100).toFixed(0)}%"></div></div>
            <span class="val">${v.toFixed(2)}</span></div>`).join("")
      : '<div class="muted">所有信号均为 0（极简请求）</div>'}
  </div>`;
}

/* ---------------- Live Chat ---------------- */

let chatSession = [];
let chatBusy = false;

/* Render LLM output as markdown; the emitted HTML is sanitised so model
 * output cannot inject scripts into the console page. */
function sanitizeHtml(html) {
  const tpl = document.createElement("template");
  tpl.innerHTML = html;
  tpl.content.querySelectorAll("script, iframe, object, embed, style, link, meta").forEach((el) => el.remove());
  tpl.content.querySelectorAll("*").forEach((el) => {
    [...el.attributes].forEach((attr) => {
      if (/^on/i.test(attr.name)) el.removeAttribute(attr.name);
      if (attr.name === "href" && /^\s*javascript:/i.test(attr.value)) el.remove();
    });
  });
  return tpl.innerHTML;
}

function renderMarkdown(text) {
  if (window.marked) {
    return sanitizeHtml(marked.parse(String(text ?? ""), { breaks: true, gfm: true }));
  }
  return `<pre class="md-fallback">${esc(text)}</pre>`;
}

function appendChatMessage(role, text, typing) {
  const log = $("#chat-log");
  const empty = log.querySelector(".empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "assistant") {
    div.innerHTML = '<div class="bubble md"></div>';
    const bubble = div.querySelector(".bubble");
    bubble.dataset.content = "";
    if (typing) bubble.innerHTML = '<span class="typing">▍</span>';
    else bubble.innerHTML = renderMarkdown(text);
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return bubble;
  }
  div.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div.querySelector(".bubble");
}

/* Incremental markdown update while the stream is live. */
function updateTyping(bubble, content) {
  bubble.dataset.content = content;
  bubble.innerHTML = renderMarkdown(content) + '<span class="typing">▍</span>';
  const log = $("#chat-log");
  log.scrollTop = log.scrollHeight;
}

/* Parse an OpenAI SSE stream from the gateway, invoking onDelta per chunk. */
async function streamChat(messages, onDelta) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { ...tenantHeaders(), "content-type": "application/json" },
    body: JSON.stringify({ model: "auto", stream: true, messages }),
  });
  if (!res.ok || !res.body) {
    let msg = `HTTP ${res.status}`;
    try { const body = await res.json(); msg = (body.error && body.error.message) || msg; } catch { /* non-JSON */ }
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      try {
        const evt = JSON.parse(data);
        if (evt.error) throw new Error(evt.error.message || JSON.stringify(evt.error));
        const delta = evt.choices && evt.choices[0] && evt.choices[0].delta;
        if (delta && typeof delta.content === "string") onDelta(delta.content);
      } catch (err) {
        if (err instanceof SyntaxError) continue; /* ignore non-JSON keep-alives */
        throw err;
      }
    }
  }
}

async function explainChat(messages) {
  const box = $("#chat-decision");
  if (!store.tenant) {
    box.innerHTML = '<div class="empty">需要配置租户 API key。</div>';
    return;
  }
  box.innerHTML = '<div class="empty">正在分析本轮路由…</div>';
  try {
    const data = await fetchJSON("/v1/routing/explain", {
      method: "POST",
      headers: { ...tenantHeaders(), "content-type": "application/json" },
      body: JSON.stringify({ model: "auto", messages }),
    });
    box.innerHTML = decisionHtml(data);
  } catch (e) {
    box.innerHTML = `<div class="error-box">分析失败：${esc(e.message)}</div>`;
  }
}

async function sendChatMessage() {
  const input = $("#chat-text");
  const text = input.value.trim();
  if (!text || chatBusy) return;
  if (!store.tenant) {
    $("#chat-decision").innerHTML = '<div class="empty">需要配置租户 API key。</div>';
    return;
  }

  const userMsg = { role: "user", content: text };
  chatSession.push(userMsg);
  appendChatMessage("user", text);
  input.value = "";
  chatBusy = true;
  $("#chat-send").disabled = true;
  $("#chat-send").textContent = "…";

  /* The routing decision is evaluated on the conversation as sent (the user
   * message is already in chatSession; the reply is not yet). */
  explainChat(chatSession);

  const bubble = appendChatMessage("assistant", "", true);
  try {
    let content = "";
    await streamChat(chatSession, (delta) => {
      content += delta;
      updateTyping(bubble, content);
    });
    bubble.dataset.content = content;
    bubble.innerHTML = renderMarkdown(content);
    chatSession.push({ role: "assistant", content });
  } catch (e) {
    bubble.innerHTML = `<div class="error-box">对话失败：${esc(e.message)}</div>`;
  } finally {
    chatBusy = false;
    $("#chat-send").disabled = false;
    $("#chat-send").textContent = "发送";
    const log = $("#chat-log");
    log.scrollTop = log.scrollHeight;
  }
}

function clearChat() {
  chatSession = [];
  $("#chat-log").innerHTML = '<div class="empty">发送一条消息开始对话，右侧会展示本轮的完整路由决策。</div>';
  $("#chat-decision").innerHTML = '<div class="empty">等待对话…</div>';
}

/* ---------------- Settings ---------------- */

const TIERS = ["economy", "standard", "premium", "reasoning"];
let cfgData = null;

async function loadSettings() {
  const status = $("#cfg-status");
  if (!store.admin) {
    status.textContent = "需要 admin key。";
    return;
  }
  try {
    const data = await fetchJSON("/admin/config", { headers: adminHeaders() });
    cfgData = data;
    renderProviderRows(data.providers || []);
    renderModelRows(data.models || [], data.providers || []);
    status.textContent = "已加载";
  } catch (e) {
    status.textContent = "加载失败";
    $("#provider-rows").innerHTML = `<div class="error-box">加载失败：${esc(e.message)}</div>`;
  }
}

function renderProviderRows(providers) {
  $("#provider-rows").innerHTML = `
    <div class="cfg-grid cfg-grid-provider">
      <span class="cfg-th">名称</span><span class="cfg-th">Base URL</span><span class="cfg-th">API Key（留空不变）</span><span></span>
      ${providers.map((p, i) => `
        <input class="cfg-name" data-i="${i}" value="${esc(p.name)}" placeholder="provider 名">
        <input class="cfg-url" data-i="${i}" value="${esc(p.base_url || "")}" placeholder="https://api.xxx.com/v1">
        <input class="cfg-key" data-i="${i}" type="password" value="" placeholder="${p.api_key ? "已配置，留空保持不变" : "api key"}">
        <button class="btn small ghost row-del" data-i="${i}" title="删除">✕</button>`).join("")}
    </div>`;
}

function renderModelRows(models, providers) {
  const names = providers.map((p) => p.name);
  $("#model-rows").innerHTML = `
    <div class="cfg-grid cfg-grid-model">
      <span class="cfg-th">ID</span><span class="cfg-th">Provider</span><span class="cfg-th">上游模型</span>
      <span class="cfg-th">档位</span><span class="cfg-th">输入$/1k</span><span class="cfg-th">输出$/1k</span>
      <span class="cfg-th">上下文</span><span class="cfg-th">质量</span><span class="cfg-th">延迟ms</span><span></span>
      ${models.map((m, i) => `
        <input class="m-id" data-i="${i}" value="${esc(m.id)}" placeholder="路由 id">
        <select class="m-provider" data-i="${i}">${names.map((n) => `<option ${n === m.provider ? "selected" : ""}>${esc(n)}</option>`).join("")}</select>
        <input class="m-upstream" data-i="${i}" value="${esc(m.upstream_model || "")}" placeholder="上游模型名">
        <select class="m-tier" data-i="${i}">${TIERS.map((t) => `<option ${t === m.tier ? "selected" : ""}>${t}</option>`).join("")}</select>
        <input class="m-in" data-i="${i}" type="number" step="0.0001" min="0" value="${m.input_cost_per_1k ?? 0}" title="输入价格（$/1k tokens）">
        <input class="m-out" data-i="${i}" type="number" step="0.001" min="0" value="${m.output_cost_per_1k ?? 0}" title="输出价格（$/1k tokens）">
        <input class="m-ctx" data-i="${i}" type="number" step="1000" min="1000" value="${m.context_window ?? 128000}" title="上下文窗口">
        <input class="m-quality" data-i="${i}" type="number" step="0.05" min="0" max="1" value="${m.quality_prior ?? 0.5}" title="质量先验 [0,1]">
        <input class="m-latency" data-i="${i}" type="number" step="100" min="1" value="${m.latency_prior_ms ?? 2000}" title="延迟先验（ms）">
        <button class="btn small ghost row-del" data-i="${i}" title="删除">✕</button>`).join("")}
    </div>`;
}

function collectProviders() {
  const grid = document.querySelector("#provider-rows .cfg-grid-provider");
  if (!grid) return [];
  const names = grid.querySelectorAll(".cfg-name");
  const urls = grid.querySelectorAll(".cfg-url");
  const keys = grid.querySelectorAll(".cfg-key");
  const out = [];
  names.forEach((el, i) => {
    const name = el.value.trim();
    const url = urls[i] ? urls[i].value.trim() : "";
    if (!name || !url) return;
    const item = { name, base_url: url };
    const key = keys[i] ? keys[i].value.trim() : "";
    if (key) item.api_key = key;
    out.push(item);
  });
  return out;
}

function collectModels() {
  const grid = document.querySelector("#model-rows .cfg-grid-model");
  if (!grid) return [];
  const ids = grid.querySelectorAll(".m-id");
  const providers = grid.querySelectorAll(".m-provider");
  const upstreams = grid.querySelectorAll(".m-upstream");
  const tiers = grid.querySelectorAll(".m-tier");
  const ins = grid.querySelectorAll(".m-in");
  const outs = grid.querySelectorAll(".m-out");
  const ctxs = grid.querySelectorAll(".m-ctx");
  const qualities = grid.querySelectorAll(".m-quality");
  const latencies = grid.querySelectorAll(".m-latency");
  const out = [];
  ids.forEach((el, i) => {
    const id = el.value.trim();
    const provider = providers[i] ? providers[i].value : "";
    if (!id || !provider) return;
    out.push({
      id,
      provider,
      upstream_model: upstreams[i] ? upstreams[i].value.trim() : "",
      tier: tiers[i] ? tiers[i].value : "standard",
      input_cost_per_1k: ins[i] ? parseFloat(ins[i].value) || 0 : 0,
      output_cost_per_1k: outs[i] ? parseFloat(outs[i].value) || 0 : 0,
      context_window: ctxs[i] ? parseInt(ctxs[i].value, 10) || 128000 : 128000,
      quality_prior: qualities[i] ? parseFloat(qualities[i].value) || 0.5 : 0.5,
      latency_prior_ms: latencies[i] ? parseFloat(latencies[i].value) || 2000 : 2000,
    });
  });
  return out;
}

function deleteProviderRow(i) {
  const providers = collectProviders();
  providers.splice(i, 1);
  renderProviderRows(providers);
  renderModelRows(collectModels(), providers);
}

function deleteModelRow(i) {
  const models = collectModels();
  models.splice(i, 1);
  renderModelRows(models, collectProviders());
}

async function saveSettings() {
  const status = $("#cfg-saved");
  if (!store.admin) return;
  status.classList.remove("hidden");
  status.textContent = "保存中…";
  try {
    const body = { providers: collectProviders(), models: collectModels() };
    await fetchJSON("/admin/config", {
      method: "PUT",
      headers: { ...adminHeaders(), "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    status.textContent = "已保存并热重载 ✓";
    setTimeout(() => status.classList.add("hidden"), 2500);
    loadSettings();
    loadModelsIntoSelect();
  } catch (e) {
    status.textContent = `保存失败：${e.message}`;
  }
}


function switchView(view) {
  currentView = view;
  document.querySelectorAll(".nav-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view));
  $("#view-dashboard").classList.toggle("hidden", view !== "dashboard");
  $("#view-chat").classList.toggle("hidden", view !== "chat");
  $("#view-playground").classList.toggle("hidden", view !== "playground");
  $("#view-settings").classList.toggle("hidden", view !== "settings");
  if (view === "dashboard") loadDashboard();
  if (view === "settings") loadSettings();
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

  // --- chat wiring ---
  $("#chat-send").addEventListener("click", sendChatMessage);
  $("#chat-text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
  $("#chat-clear").addEventListener("click", clearChat);

  // --- settings wiring ---
  $("#add-provider").addEventListener("click", () => {
    const providers = collectProviders();
    providers.push({ name: "", base_url: "", api_key: "" });
    renderProviderRows(providers);
  });
  $("#add-model").addEventListener("click", () => {
    const models = collectModels();
    models.push({
      id: "", provider: "", upstream_model: "", tier: "standard",
      input_cost_per_1k: 0, output_cost_per_1k: 0, context_window: 128000,
      quality_prior: 0.5, latency_prior_ms: 2000,
    });
    renderModelRows(models, collectProviders());
  });
  $("#save-config").addEventListener("click", saveSettings);
  $("#provider-rows").addEventListener("click", (e) => {
    const btn = e.target.closest(".row-del");
    if (btn) deleteProviderRow(parseInt(btn.dataset.i, 10));
  });
  $("#model-rows").addEventListener("click", (e) => {
    const btn = e.target.closest(".row-del");
    if (btn) deleteModelRow(parseInt(btn.dataset.i, 10));
  });

  renderLogin();
  startPoll();
  ping();
  setInterval(ping, 15000);
  switchView("dashboard");
  loadModelsIntoSelect();
}

document.addEventListener("DOMContentLoaded", init);
