/* Remine Workbench Client */
(() => {
  const state = {
    snapshots: [],
    runData: null,
    selectedBriefId: null,
    benchmarks: null,
    rawTokenView: {},
    selectedTokenByBrief: {},
  };

  const els = {
    snapshotSelect: document.getElementById("snapshot-select"),
    modeSelect: document.getElementById("mode-select"),
    modelSelect: document.getElementById("model-select"),
    gate0Toggle: document.getElementById("gate0-toggle"),
    runBtn: document.getElementById("run-btn"),
    harvestDate: document.getElementById("harvest-date"),
    harvestPid: document.getElementById("harvest-pid"),
    harvestBtn: document.getElementById("harvest-btn"),
    statusMsg: document.getElementById("status-msg"),
    mcpPill: document.getElementById("mcp-pill"),
    llmPill: document.getElementById("llm-pill"),
    releaseBanner: document.getElementById("release-banner"),
    kpiStrip: document.getElementById("kpi-strip"),
    briefPills: document.getElementById("brief-pills"),
    graphSvgWrap: document.getElementById("graph-svg-wrap"),
    traceSubtitle: document.getElementById("trace-subtitle"),
    traceDetail: document.getElementById("trace-detail"),
    tabStories: document.getElementById("tab-stories"),
    tabLedger: document.getElementById("tab-ledger"),
    tabReview: document.getElementById("tab-review"),
    tabBenchmarks: document.getElementById("tab-benchmarks"),
  };

  function esc(val) {
    return String(val ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function setStatus(text) {
    els.statusMsg.textContent = text;
  }

  function verdictClass(verdict) {
    return `verdict-${String(verdict || "skip").toLowerCase()}`;
  }

  async function apiJson(url, options = {}) {
    const resp = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  async function refreshHealthAndSnapshots(preferredSnapshotId = null) {
    const [health, snapResp, benchResp] = await Promise.all([
      apiJson("/api/health"),
      apiJson("/api/snapshots"),
      apiJson("/api/benchmarks"),
    ]);

    const toolCount = (health.mcp_tools || []).length;
    els.mcpPill.textContent = `MCP: ${toolCount} tools active`;
    els.mcpPill.className = "pill pill-ok";

    const llm = health.local_llm || {};
    if (llm.online) {
      els.llmPill.textContent = `Local /v1: ${(llm.models || []).length} model(s)`;
      els.llmPill.className = "pill pill-ok";
      const existing = Array.from(els.modelSelect.options).map((o) => o.value);
      for (const m of llm.models || []) {
        if (!existing.includes(m)) {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          els.modelSelect.appendChild(opt);
        }
      }
    } else {
      els.llmPill.textContent = "Local /v1: offline (Replay ready)";
      els.llmPill.className = "pill pill-warn";
    }

    state.snapshots = snapResp.snapshots || [];
    state.benchmarks = benchResp;
    const current = preferredSnapshotId || els.snapshotSelect.value;
    els.snapshotSelect.innerHTML = state.snapshots
      .map(
        (s) =>
          `<option value="${esc(s.id)}">[${esc(s.source)}] ${esc(s.title)} (${esc(s.brief_count)} briefs)</option>`
      )
      .join("");
    if (current && state.snapshots.some((s) => s.id === current)) {
      els.snapshotSelect.value = current;
    }
  }

  function renderReleaseAndKpis() {
    const data = state.runData;
    if (!data) return;
    const rel = data.release || {};
    const cfg = data.config || {};
    const sum = data.summary || {};

    els.releaseBanner.innerHTML = `
      <div class="release-top">
        <h2>${esc(rel.title)}</h2>
        <div class="status-pills">
          <span class="badge-mono">Release: ${esc(rel.id)}</span>
          <span class="badge-mono">Model: ${esc(cfg.model)}</span>
          <span class="badge-mono">Gate 0: ${cfg.enforce_semantic_standardization ? "STRICT" : "OFF"}</span>
          ${rel.table_url ? `<a class="btn btn-ghost" href="${esc(rel.table_url)}" target="_blank" rel="noopener">StatCan Cube</a>` : ""}
        </div>
      </div>
      <p class="release-highlights">${esc(rel.highlights)}</p>
    `;

    const evalObj = sum.evaluation;
    const kpis = [
      { label: "Published Stories", value: sum.publish_count ?? 0 },
      { label: "Skipped (Novelty / Confounded)", value: sum.skip_count ?? 0 },
      { label: "Human Review Queue", value: sum.review_count ?? 0 },
      { label: "Unreliable Blocked", value: sum.unreliable_count ?? 0 },
      { label: "LLM Agent Calls", value: sum.llm_calls ?? 0 },
      { label: "MCP Tool Calls", value: sum.mcp_calls ?? 0 },
      ...(evalObj
        ? [{ label: "Benchmark Verdict Acc.", value: `${Math.round(evalObj.verdict_accuracy * 100)}%` }]
        : []),
    ];

    els.kpiStrip.innerHTML = kpis
      .map(
        (k) => `
          <div class="kpi-card">
            <div class="kpi-label">${esc(k.label)}</div>
            <div class="kpi-value">${esc(k.value)}</div>
          </div>
        `
      )
      .join("");
  }

  function renderPipelineGraph() {
    const data = state.runData;
    if (!data) return;
    const ledger = data.ledger || [];
    if (!ledger.length) return;

    if (!ledger.some((r) => r.brief_id === state.selectedBriefId)) {
      state.selectedBriefId = ledger[0].brief_id;
    }

    els.briefPills.innerHTML = ledger
      .map((row) => {
        const selected = row.brief_id === state.selectedBriefId ? "selected" : "";
        const retryTag = row.attempts > 1 ? ` · R${row.attempts}` : "";
        return `
          <button
            type="button"
            class="brief-pill ${verdictClass(row.verdict)} ${selected}"
            data-brief-id="${esc(row.brief_id)}"
          >${esc(row.brief_id)} · ${esc(row.verdict.toUpperCase())}${esc(retryTag)}</button>
        `;
      })
      .join("");

    els.briefPills.querySelectorAll(".brief-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.selectedBriefId = btn.dataset.briefId;
        renderPipelineGraph();
      });
    });

    const selectedRow = ledger.find((r) => r.brief_id === state.selectedBriefId);
    const trace = selectedRow ? selectedRow.trace || [] : [];
    const nodeStatus = {};
    let hadRetry = false;
    for (const ev of trace) {
      nodeStatus[ev.node] = ev.status;
      if (ev.status === "retry") hadRetry = true;
    }

    const nodes = [
      { id: "gate", label: "Gate 0\nReliability + Semantic", x: 18, y: 62, w: 122 },
      { id: "retrieve", label: "MCP Retrieve\nRules + Memory", x: 162, y: 62, w: 112 },
      { id: "angle", label: "Angle Selector\nLLM", x: 296, y: 62, w: 108 },
      { id: "draft", label: "Token Drafter\nLLM", x: 426, y: 62, w: 108 },
      { id: "bind_check", label: "Bind Check\nContract", x: 556, y: 62, w: 104 },
      { id: "critic", label: "Method Critic\nLLM", x: 682, y: 62, w: 104 },
      { id: "review", label: "Human Review\nEscalation", x: 380, y: 168, w: 140 },
    ];

    const styleFor = (id) => {
      const st = nodeStatus[id];
      if (!st) return { fill: "#ffffff", stroke: "#cbd5e1", text: "#64748b" };
      if (st === "blocked" || st === "escalated" || st === "error") {
        return { fill: "#fdf1f2", stroke: "#af3c43", text: "#7f1d1d" };
      }
      if (st === "retry") {
        return { fill: "#fef6e7", stroke: "#b45309", text: "#92400e" };
      }
      return { fill: "#eaf6f5", stroke: "#1d7874", text: "#0f4c49" };
    };

    const rectSvg = nodes
      .map((n) => {
        const c = styleFor(n.id);
        const lines = n.label.split("\n");
        return `
          <g>
            <rect x="${n.x}" y="${n.y}" width="${n.w}" height="52" rx="8"
              fill="${c.fill}" stroke="${c.stroke}" stroke-width="2" />
            <text x="${n.x + n.w / 2}" y="${n.y + 22}" text-anchor="middle"
              font-size="11.5" font-weight="700" fill="${c.text}">${esc(lines[0])}</text>
            <text x="${n.x + n.w / 2}" y="${n.y + 38}" text-anchor="middle"
              font-size="10" fill="${c.text}">${esc(lines[1])}</text>
          </g>
        `;
      })
      .join("");

    const retryColor = hadRetry ? "#b45309" : "#cbd5e1";
    const svg = `
      <svg viewBox="0 0 806 238" width="100%" height="225" role="img" aria-label="LangGraph workflow graph">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#64748b" />
          </marker>
          <marker id="arrow-retry" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="${retryColor}" />
          </marker>
        </defs>
        <line x1="140" y1="88" x2="162" y2="88" stroke="#64748b" stroke-width="1.8" marker-end="url(#arrow)" />
        <line x1="274" y1="88" x2="296" y2="88" stroke="#64748b" stroke-width="1.8" marker-end="url(#arrow)" />
        <line x1="404" y1="88" x2="426" y2="88" stroke="#64748b" stroke-width="1.8" marker-end="url(#arrow)" />
        <line x1="534" y1="88" x2="556" y2="88" stroke="#64748b" stroke-width="1.8" marker-end="url(#arrow)" />
        <line x1="660" y1="88" x2="682" y2="88" stroke="#64748b" stroke-width="1.8" marker-end="url(#arrow)" />

        <!-- Bounded retry loops -->
        <path d="M 608 62 C 608 24, 480 24, 480 62" fill="none" stroke="${retryColor}"
          stroke-width="2" stroke-dasharray="4 3" marker-end="url(#arrow-retry)" />
        <path d="M 734 62 C 734 14, 468 14, 468 62" fill="none" stroke="${retryColor}"
          stroke-width="2" stroke-dasharray="4 3" marker-end="url(#arrow-retry)" />
        <text x="580" y="18" text-anchor="middle" font-size="10" fill="${retryColor}" font-weight="600">Bounded Retry Loop (attempt &lt;= max_attempts)</text>

        <!-- Escalation edges to review -->
        <path d="M 608 114 L 500 168" fill="none" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="3 3" marker-end="url(#arrow)" />
        <path d="M 734 114 L 520 185" fill="none" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="3 3" marker-end="url(#arrow)" />
        <path d="M 350 114 L 405 168" fill="none" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="3 3" marker-end="url(#arrow)" />

        ${rectSvg}
      </svg>
    `;
    els.graphSvgWrap.innerHTML = svg;
    renderTracePanel(selectedRow);
  }

  function renderTracePanel(row) {
    if (!row) {
      els.traceDetail.innerHTML = "";
      return;
    }
    els.traceSubtitle.textContent = `${row.brief_id} · ${row.title} (${row.verdict.toUpperCase()})`;

    const traceHtml = (row.trace || [])
      .map(
        (ev) => `
          <div class="trace-item">
            <div class="trace-item-head">
              <span>${esc(ev.node)} · ${esc(ev.status)} ${ev.attempt ? `(attempt ${esc(ev.attempt)})` : ""}</span>
              <span>${esc(ev.duration_ms)} ms</span>
            </div>
            <p>${esc(ev.summary)}</p>
          </div>
        `
      )
      .join("");

    const mcpHtml = (row.mcp_calls || []).length
      ? `
        <div class="trace-item">
          <div class="trace-item-head">
            <span>MCP JSON-RPC 2.0 Calls (${row.mcp_calls.length})</span>
            <span>Audited</span>
          </div>
          <p class="mono">${row.mcp_calls.map((c) => `${esc(c.tool)}(${esc(JSON.stringify(c.arguments))}) -> ${esc(c.duration_ms)}ms`).join("<br>")}</p>
        </div>
      `
      : "";

    const feedbackHtml = (row.feedback_log || []).length
      ? `
        <div class="trace-item" style="border-color:#f3d7a4;background:#fef6e7;">
          <div class="trace-item-head"><span>Self-Correction Feedback Log</span><span>${row.feedback_log.length} note(s)</span></div>
          <p>${row.feedback_log.map((f) => `• ${esc(f)}`).join("<br>")}</p>
        </div>
      `
      : "";

    els.traceDetail.innerHTML = traceHtml + mcpHtml + feedbackHtml;
  }

  function renderBoundParagraph(story, showRaw) {
    const segs = story.body_segments || [];
    if (!segs.length) return esc(showRaw ? story.raw_body || story.body : story.body);
    return segs
      .map((seg) => {
        if (seg.kind === "text") return esc(seg.text);
        const label = showRaw ? seg.token : seg.text;
        const rawCls = showRaw ? "raw" : "";
        return `<span class="token-span ${rawCls}" data-brief="${esc(story.brief_id)}" data-fact="${esc(seg.fact_id)}" title="Click to inspect StatCan vector provenance">${esc(label)}</span>`;
      })
      .join("");
  }

  function renderStoriesTab() {
    const stories = (state.runData && state.runData.stories) || [];
    if (!stories.length) {
      els.tabStories.innerHTML = `<div class="card"><p class="muted">No stories were published in this run (check the Audit Ledger or Human Review tabs).</p></div>`;
      return;
    }

    els.tabStories.innerHTML = stories
      .map((story) => {
        const showRaw = Boolean(state.rawTokenView[story.brief_id]);
        const activeFactId =
          state.selectedTokenByBrief[story.brief_id] ||
          ((story.token_bindings || [])[0] && story.token_bindings[0].fact_id);
        const activeBinding = (story.token_bindings || []).find((b) => b.fact_id === activeFactId);

        const drawerHtml = activeBinding
          ? `
            <div class="token-drawer">
              <strong>Token Provenance Inspector:</strong>
              <span class="mono">${esc(activeBinding.token)}</span> &rarr;
              <span class="mono">${esc(activeBinding.resolved_text)}</span>
              <div class="mono" style="margin-top:4px;color:#334155;">
                Vectors: ${esc((activeBinding.vectors || []).join(", "))} |
                Periods: ${esc((activeBinding.periods || []).join(", "))} |
                Cut: ${esc(JSON.stringify(activeBinding.cut))} |
                Held At: ${esc(JSON.stringify(activeBinding.held_at))}
              </div>
            </div>
          `
          : "";

        return `
          <article class="card story-card">
            <div class="story-header">
              <div>
                <div class="status-pills">
                  <span class="badge-mono">${esc(story.brief_id)}</span>
                  <span class="badge-mono">${esc(story.stance.toUpperCase())}</span>
                  <span class="badge-mono">Attempts: ${esc(story.attempts)}${story.recovered_via_retry ? " (RECOVERED)" : ""}</span>
                </div>
                <h3 class="story-headline">${esc(story.headline)}</h3>
              </div>
              <button class="btn btn-ghost toggle-token-btn" type="button" data-brief="${esc(story.brief_id)}">
                ${showRaw ? "Show Bound Prose" : "Show {{token}} Overlay"}
              </button>
            </div>

            <p class="story-body">${renderBoundParagraph(story, showRaw)}</p>
            ${drawerHtml}

            <div class="story-meta-grid">
              <div class="meta-box">
                <h4>Explicit Assumption</h4>
                <div>${esc(story.assumption)}</div>
              </div>
              <div class="meta-box">
                <h4>Novelty vs. The Daily</h4>
                <div>${esc(story.differs_from_daily)}</div>
              </div>
            </div>

            <div class="status-pills">
              <span class="pill pill-neutral">Citations: ${esc((story.citations || []).join(", "))}</span>
              <span class="pill pill-ok">Critic: ${esc(story.critic_note)}</span>
              ${story.table_url ? `<a class="btn btn-ghost" href="${esc(story.table_url)}" target="_blank" rel="noopener">Open StatCan Cube</a>` : ""}
            </div>
          </article>
        `;
      })
      .join("");

    els.tabStories.querySelectorAll(".toggle-token-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.brief;
        state.rawTokenView[id] = !state.rawTokenView[id];
        renderStoriesTab();
      });
    });

    els.tabStories.querySelectorAll(".token-span").forEach((span) => {
      span.addEventListener("click", () => {
        state.selectedTokenByBrief[span.dataset.brief] = span.dataset.fact;
        renderStoriesTab();
      });
    });
  }

  function renderLedgerTab() {
    const data = state.runData;
    if (!data) return;
    const manifest = data.coverage_manifest;
    const ledger = data.ledger || [];

    const manifestCard = manifest
      ? `
        <div class="card">
          <div class="section-head">
            <div>
              <h2>Two-Tier Coverage Subtraction Manifest</h2>
              <p class="muted">Tier 1 (The Daily release) and Tier 2 (Analytical Study ${esc(manifest.paper_catalogue || "None")}) claimed coordinate envelopes.</p>
            </div>
            <span class="badge-mono">${esc((manifest.claimed_coordinates || []).length)} Claimed Cut(s)</span>
          </div>
          <table>
            <thead><tr><th>Tier</th><th>Period</th><th>Claimed Dimensions</th><th>Published Quote</th></tr></thead>
            <tbody>
              ${(manifest.claimed_coordinates || [])
                .map(
                  (c) => `
                    <tr>
                      <td><span class="badge-mono">${esc(c.tier)}</span></td>
                      <td class="mono">${esc(c.period || "—")}</td>
                      <td class="mono">${esc(JSON.stringify(c.dimensions))}</td>
                      <td>${esc(c.prose_quote)}</td>
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      `
      : "";

    const ledgerCard = `
      <div class="card">
        <div class="section-head">
          <div>
            <h2>Full Editorial Audit Ledger</h2>
            <p class="muted">Every candidate brief evaluated with deterministic quality gates, MCP rule citations, and novelty overlap scores.</p>
          </div>
        </div>
        <table>
          <thead><tr><th>Brief</th><th>Verdict</th><th>Category / Gate</th><th>Overlaps</th><th>Reason &amp; Citations</th></tr></thead>
          <tbody>
            ${ledger
              .map(
                (row) => `
                  <tr>
                    <td><strong>${esc(row.brief_id)}</strong><br><span class="muted">${esc(row.title)}</span></td>
                    <td><span class="brief-pill ${verdictClass(row.verdict)}">${esc(row.verdict.toUpperCase())}</span></td>
                    <td class="mono">${esc(row.rejection_category || (row.reliable ? `flag=${row.quality_flag}` : `BLOCKED (${row.quality_flag})`))}</td>
                    <td class="mono">${(row.coverage_overlaps || []).length ? row.coverage_overlaps.map((o) => `${esc(o.tier)} (${Math.round(o.overlap_ratio * 100)}%)`).join("<br>") : "0 overlaps"}</td>
                    <td>${esc(row.reason)}<div class="mono muted" style="margin-top:4px;">Citations: ${esc((row.citations || []).join(", "))}</div></td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;

    els.tabLedger.innerHTML = manifestCard + ledgerCard;
  }

  function renderReviewTab() {
    const queue = (state.runData && state.runData.review_queue) || [];
    if (!queue.length) {
      els.tabReview.innerHTML = `<div class="card"><p class="muted">Zero briefs currently require human editorial escalation.</p></div>`;
      return;
    }

    els.tabReview.innerHTML = queue
      .map(
        (item) => `
          <div class="card" style="border-left:4px solid var(--amber);">
            <div class="story-header">
              <div>
                <span class="brief-pill ${verdictClass(item.verdict)}">${esc(item.verdict.toUpperCase())}</span>
                <strong style="margin-left:8px;">${esc(item.brief_id)} — ${esc(item.title)}</strong>
              </div>
              <span class="badge-mono">Flag: ${esc(item.quality_flag)} | Attempts: ${esc(item.attempts)}</span>
            </div>
            <p>${esc(item.reason)}</p>
            ${item.critic_note ? `<div class="meta-box"><h4>Escalation Note</h4><div class="mono">${esc(item.critic_note)}</div></div>` : ""}
            ${(item.feedback_log || []).length ? `<div class="meta-box" style="margin-top:8px;"><h4>Retry Trace</h4><div class="mono">${item.feedback_log.map(esc).join("<br>")}</div></div>` : ""}
          </div>
        `
      )
      .join("");
  }

  function renderBenchmarksTab() {
    const evalObj = state.runData && state.runData.summary && state.runData.summary.evaluation;
    const runs = (state.benchmarks && state.benchmarks.runs) || [];

    const currentCard = evalObj
      ? `
        <div class="card">
          <h2>Current Run Evaluation (${esc(state.runData.config.model)})</h2>
          <p class="muted">Verdict Accuracy: <strong>${Math.round(evalObj.verdict_accuracy * 100)}%</strong> (${esc(evalObj.correct_verdicts)}/${esc(evalObj.total)}) · Skip Category Accuracy: <strong>${Math.round(evalObj.rejection_accuracy * 100)}%</strong> · Self-Correction Recoveries: <strong>${esc(evalObj.recovered_via_retry)}</strong></p>
          <pre class="mono" style="background:#faf8f4;padding:10px;border-radius:6px;overflow:auto;">Confusion Matrix: ${esc(JSON.stringify(evalObj.confusion, null, 2))}</pre>
        </div>
      `
      : "";

    const benchCard = runs.length
      ? `
        <div class="card">
          <div class="section-head">
            <div>
              <h2>Multi-Model &amp; Gate 0 Ablation Benchmarks</h2>
              <p class="muted">Measured locally on Apple Silicon /v1 across deterministic Replay and local open-weight models.</p>
            </div>
            <span class="badge-mono">${runs.length} Benchmark Runs</span>
          </div>
          <table>
            <thead><tr><th>Dataset</th><th>Model</th><th>Gate 0</th><th>Verdict Acc.</th><th>Rejection Acc.</th><th>LLM Calls</th><th>Duration</th></tr></thead>
            <tbody>
              ${runs
                .map((r) => {
                  const pct = Math.round((r.verdict_accuracy || 0) * 100);
                  const rej = Math.round((r.rejection_accuracy || 0) * 100);
                  return `
                    <tr>
                      <td>${esc(r.dataset)}</td>
                      <td class="mono">${esc(r.model)}</td>
                      <td class="mono">${r.gate0 ? "STRICT" : "OFF"}</td>
                      <td>
                        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                        <span class="mono">${pct}%</span>
                      </td>
                      <td class="mono">${rej}%</td>
                      <td class="mono">${esc(r.llm_calls)}</td>
                      <td class="mono">${esc(r.elapsed_s)}s</td>
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      `
      : `<div class="card"><p class="muted">No saved multi-model benchmark matrix found yet.</p></div>`;

    els.tabBenchmarks.innerHTML = currentCard + benchCard;
  }

  function renderAll() {
    renderReleaseAndKpis();
    renderPipelineGraph();
    renderStoriesTab();
    renderLedgerTab();
    renderReviewTab();
    renderBenchmarksTab();
  }

  async function executeRun() {
    els.runBtn.disabled = true;
    setStatus("Running LangGraph swarm...");
    try {
      const payload = {
        snapshot_id: els.snapshotSelect.value,
        mode: els.modeSelect.value,
        model: els.modelSelect.value,
        enforce_semantic_standardization: els.gate0Toggle.checked,
      };
      state.runData = await apiJson("/api/run", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderAll();
      setStatus(
        `Completed in ${state.runData.summary.total_duration_ms} ms (${state.runData.summary.llm_calls} LLM / ${state.runData.summary.mcp_calls} MCP calls)`
      );
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      els.runBtn.disabled = false;
    }
  }

  async function executeHarvest() {
    els.harvestBtn.disabled = true;
    setStatus("Harvesting live StatCan Daily article + WDS cube slices...");
    try {
      const body = {
        date: els.harvestDate.value.trim() || null,
        pid: els.harvestPid.value.trim() || null,
      };
      const resp = await apiJson("/api/harvest", {
        method: "POST",
        body: JSON.stringify(body),
      });
      await refreshHealthAndSnapshots(resp.snapshot_id);
      if (resp.snapshot_id) {
        els.snapshotSelect.value = resp.snapshot_id;
      }
      setStatus(`Harvested PID ${resp.harvested.pid} (${resp.harvested.brief_count} briefs). Running swarm...`);
      if (els.modeSelect.value === "replay") {
        // Fresh live harvests do not have synthetic replay entries; switch cleanly if local /v1 is online.
        if (els.llmPill.className.includes("pill-ok")) {
          els.modeSelect.value = "live";
        }
      }
      if (els.modeSelect.value === "live") {
        await executeRun();
      }
    } catch (err) {
      setStatus(`Harvest failed: ${err.message}`);
    } finally {
      els.harvestBtn.disabled = false;
    }
  }

  function wireTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      });
    });
  }

  async function init() {
    wireTabs();
    els.runBtn.addEventListener("click", executeRun);
    els.harvestBtn.addEventListener("click", executeHarvest);
    await refreshHealthAndSnapshots();
    await executeRun();
  }

  init().catch((err) => setStatus(`Initialization error: ${err.message}`));
})();
