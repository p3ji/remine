"use strict";

const state = {
  runtimeMode: "detecting", // "server" | "static"
  health: null,
  staticBundle: null,
  snapshots: [],
  benchmarks: { runs: [] },
  activeRun: null,
  selectedToken: null,
  selectedStoryId: null,
  highlightsExpanded: false,
};

const PROCESS_STEPS = [
  {
    num: 1,
    title: "Data Reliability Check",
    desc: "Automatically holds back Statistics Canada estimates flagged as suppressed or too unreliable to publish (such as Flag F or unavailable cells).",
  },
  {
    num: 2,
    title: "Fair Comparison Guard",
    desc: "Checks whether comparing two regions or sectors holds key factors steady (for example, comparing core-aged workers rather than unadjusted all-age totals).",
  },
  {
    num: 3,
    title: "Coverage & Rules Lookup",
    desc: "Checks whether Statistics Canada already reported the finding in The Daily or a companion paper, and retrieves official table methodology notes.",
  },
  {
    num: 4,
    title: "Plain-English Draft with Locked Figures",
    desc: "Writes a short policy summary where every statistic must come verbatim from a verified Statistics Canada table slice rather than free-text generation.",
  },
  {
    num: 5,
    title: "Independent Quality Review",
    desc: "A second check confirms all qualifiers, time periods, and comparison caveats are stated accurately—or sends the draft back for revision.",
  },
];

const qs = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function showError(msg) {
  const el = qs("error-banner");
  if (!msg) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.textContent = msg;
  el.hidden = false;
}

function formatDateLabel(dateStr) {
  if (!dateStr) return "";
  const match = String(dateStr).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return dateStr;
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return date.toLocaleDateString("en-CA", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatPeriodList(periods) {
  if (!Array.isArray(periods) || periods.length === 0) return "Not specified";
  if (periods.length === 2) return `${periods[0]} to ${periods[1]}`;
  return periods.join(", ");
}

function humanizeKey(rawKey) {
  return String(rawKey || "")
    .replace(/_high$/i, " (Higher)")
    .replace(/_low$/i, " (Lower)")
    .replace(/_/g, " ");
}

function renderKvChips(obj) {
  if (!obj || typeof obj !== "object") return "<span class=\"muted-note\">None</span>";
  const entries = Object.entries(obj);
  if (entries.length === 0) return "<span class=\"muted-note\">None</span>";
  return `<div class="kv-chip-list">${entries
    .map(
      ([k, v]) =>
        `<span class="kv-chip"><strong>${escapeHtml(humanizeKey(k))}:</strong> ${escapeHtml(v)}</span>`
    )
    .join("")}</div>`;
}

function formatKvPlainText(obj) {
  if (!obj || typeof obj !== "object") return "";
  return Object.entries(obj)
    .map(([k, v]) => `${humanizeKey(k)}: ${v}`)
    .join("; ");
}

function formatQualityFlag(flag) {
  const map = {
    A: "Very reliable (StatCan Flag A)",
    B: "Reliable (StatCan Flag B)",
    C: "Good quality (StatCan Flag C)",
    D: "Acceptable quality (StatCan Flag D)",
    E: "Use with caution (StatCan Flag E)",
    F: "Too unreliable to publish (StatCan Flag F)",
    "..": "Not available / suppressed (StatCan Flag ..)",
    x: "Suppressed for confidentiality (Flag x)",
  };
  return map[flag] || `StatCan Quality Flag ${flag || "A"}`;
}

function formatStanceLabel(stance) {
  const map = {
    concretizes: "Adds historical or subgroup detail",
    contrasts: "Highlights subgroup contrast",
    extends: "Extends published trend",
  };
  return map[stance] || "Verified table finding";
}

function formatVerdictMeta(verdict, rejectionCategory) {
  if (verdict === "publish") {
    return { label: "Ready to share", cls: "badge-teal" };
  }
  if (verdict === "unreliable_data") {
    return { label: "Held back — unreliable or suppressed", cls: "badge-crimson" };
  }
  if (verdict === "needs_editorial_review") {
    return { label: "Held back — needs analyst review", cls: "badge-amber" };
  }
  if (verdict === "skip") {
    const reasonMap = {
      ALREADY_IN_THE_DAILY: "Filtered — already in The Daily",
      ALREADY_IN_ANALYTICAL_PAPER: "Filtered — covered in StatCan report",
      CONFOUNDED_COMPARISON: "Filtered — skewed subgroup comparison",
    };
    return {
      label: reasonMap[rejectionCategory] || "Filtered out during screening",
      cls: rejectionCategory === "CONFOUNDED_COMPARISON" ? "badge-amber" : "badge-muted",
    };
  }
  return { label: verdict || "Screened", cls: "badge-muted" };
}

function formatModelLabel(modelName) {
  const map = {
    replay: "Pre-verified benchmark script",
    "qwen3.8-27b": "Qwen 3.8 (27B Local)",
    "deepseek-v4-flash@q8_0": "DeepSeek v4 Flash (Local)",
    "microsoft/phi-4": "Phi-4 (14B Local)",
  };
  return map[modelName] || modelName;
}

function formatStatCanPid(pid) {
  const digits = String(pid || "").replace(/\D/g, "");
  if (digits.length === 8) {
    return `Table ${digits.slice(0, 2)}-${digits.slice(2, 4)}-${digits.slice(4, 8)}-01`;
  }
  return pid ? `Table ${pid}` : "";
}

function formatStatCanReason(reason) {
  if (!reason) return "";
  return String(reason)
    .replace(/quality_flag='([^']+)',\s*reliable=False/gi, "Statistics Canada reliability flag '$1'")
    .replace(/quality_flag='([^']+)'/gi, "Statistics Canada reliability flag '$1'")
    .replace(/reliable=False/gi, "marked as unreliable by Statistics Canada")
    .replace(/Blocked by deterministic reliability gate/gi, "Held back by data quality check")
    .replace(/Blocked by Gate 0 semantic guard/gi, "Filtered by subgroup comparability check")
    .replace(/Blocked by metadata semantic comparability gate/gi, "Filtered by subgroup comparability check")
    .replace(/Semantic confound metadata check:\s*/gi, "")
    .replace(/\s*\(violates M\d+\)/gi, "")
    .replace(/\bM:(M\d+)\b/g, "Rule $1")
    .replace(/Methodology rule (M\d+) /gi, "Methodology check ($1) ")
    .replace(/binds all numbers from verified tokens/gi, "links all figures directly to verified source series")
    .replace(/binds all facts to tokens/gi, "links all figures directly to verified source series")
    .replace(/verified tokens/gi, "verified source series")
    .replace(/\s+([.,;])/g, "$1")
    .trim();
}

function populateSelect(selectEl, items, getValue, getLabel) {
  const prev = selectEl.value;
  selectEl.innerHTML = "";
  items.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = getValue(item);
    opt.textContent = getLabel(item);
    selectEl.appendChild(opt);
  });
  if (items.some((item) => getValue(item) === prev)) {
    selectEl.value = prev;
  }
}

function setStatusBadge() {
  const pill = qs("status-mode");
  if (state.runtimeMode === "static") {
    pill.textContent = "Published Briefs Archive";
    pill.className = "status-pill archive";
    return;
  }
  const aiOnline = Boolean(state.health?.local_llm?.online);
  pill.textContent = aiOnline
    ? "Local Server · AI Ready"
    : "Local Server · Saved Reviews Ready";
  pill.className = "status-pill online";
}

function setServerControlsVisibility() {
  const isServer = state.runtimeMode === "server";
  qs("server-run-wrap").hidden = !isServer;
  qs("advanced-local-panel").hidden = !isServer;
}

function splitParagraphsWithSegments(segments) {
  const paragraphs = [];
  let current = [];
  (segments || []).forEach((seg) => {
    if (seg.kind === "text") {
      const parts = String(seg.text || "").split(/\n\n+/);
      parts.forEach((part, idx) => {
        if (idx > 0) {
          paragraphs.push(current);
          current = [];
        }
        if (part) current.push({ kind: "text", text: part });
      });
    } else {
      current.push(seg);
    }
  });
  if (current.length > 0) paragraphs.push(current);
  return paragraphs;
}

function findRulesForBrief(briefId) {
  const ledgerRow = (state.activeRun?.ledger || []).find((r) => r.brief_id === briefId);
  return ledgerRow?.rules || [];
}

function buildCopyText(story) {
  const lines = [
    story.headline || story.title,
    "",
    story.body || "",
    "",
    `Source: Statistics Canada — ${state.activeRun?.release?.title || ""}`,
  ];
  if (story.table_url) lines.push(`Table link: ${story.table_url}`);
  const vectors = Array.from(
    new Set((story.provenance || []).flatMap((p) => p.vectors || []))
  );
  if (vectors.length > 0) lines.push(`Series vector(s): ${vectors.join(", ")}`);
  return lines.join("\n");
}

function renderReleaseAndSummary(data) {
  const release = data.release || {};
  const summary = data.summary || {};

  const parts = [];
  if (release.date) parts.push(`Released ${formatDateLabel(release.date)}`);
  if (release.pid) parts.push(formatStatCanPid(release.pid));
  qs("release-meta").textContent = parts.join(" · ") || "Statistics Canada Release";
  qs("release-title").textContent = release.title || "Selected Release";

  const dailyLink = qs("release-daily-link");
  if (release.url) {
    dailyLink.href = release.url;
    dailyLink.hidden = false;
  } else {
    dailyLink.hidden = true;
  }

  const tableLink = qs("release-table-link");
  if (release.table_url) {
    tableLink.href = release.table_url;
    tableLink.hidden = false;
  } else {
    tableLink.hidden = true;
  }

  const rawHighlights = String(release.highlights || "")
    .replace(/^The Daily\s*\|.*?Released:\s*\d{4}-\d{2}-\d{2}\s*(?:Select geography\s*)?/i, "")
    .replace(/\s*\(\s*-\s*(\d+)/g, " (-$1")
    .replace(/(\d+)\s+\.(\d+%)/g, "$1.$2")
    .replace(/-\s*(\d+)\s+(\d+\.\d+%)/g, "-$1$2")
    .trim();
  const highlightsEl = qs("release-highlights");
  const toggleBtn = qs("toggle-highlights-btn");
  if (rawHighlights.length > 360 && !state.highlightsExpanded) {
    const trimmed = rawHighlights.slice(0, 360).replace(/\s+\S*$/, "").trim();
    highlightsEl.textContent = `${trimmed}…`;
    toggleBtn.textContent = "Show full summary";
    toggleBtn.hidden = false;
  } else {
    highlightsEl.textContent = rawHighlights || "No official summary recorded for this snapshot.";
    if (rawHighlights.length > 360) {
      toggleBtn.textContent = "Show shorter summary";
      toggleBtn.hidden = false;
    } else {
      toggleBtn.hidden = true;
    }
  }

  const total = summary.total_briefs || 0;
  const heldCount = (summary.review_count || 0) + (summary.unreliable_count || 0);
  qs("kpi-published").textContent = `${summary.publish_count ?? 0} of ${total}`;
  qs("kpi-skipped").textContent = String(summary.skip_count ?? 0);
  qs("kpi-review").textContent = String(heldCount);

  const evalSummary = summary.evaluation;
  if (evalSummary && typeof evalSummary.verdict_accuracy === "number") {
    const pct = Math.round(evalSummary.verdict_accuracy * 100);
    qs("kpi-accuracy").textContent = `${pct}% (${evalSummary.correct_verdicts}/${evalSummary.total})`;
    qs("kpi-accuracy-sub").textContent = "Agreement with expert screening review";
  } else {
    qs("kpi-accuracy").textContent = `${total} checked`;
    qs("kpi-accuracy-sub").textContent = "Screened against StatCan quality & overlap rules";
  }

  qs("tab-count-findings").textContent = String((data.stories || []).length);
  qs("tab-count-checked").textContent = String((data.ledger || []).length);
  qs("tab-count-held").textContent = String((data.review_queue || []).length);
}

function renderSourceDrawer() {
  const details = qs("source-details");
  if (!state.selectedToken || !state.activeRun) {
    details.className = "source-details-empty";
    details.textContent =
      "Select any underlined figure inside a finding to verify its exact Statistics Canada series, group definitions, and time periods.";
    return;
  }

  const story = (state.activeRun.stories || []).find((s) => s.brief_id === state.selectedStoryId);
  const fact = story?.facts?.find((f) => f.id === state.selectedToken.fact_id);
  const prov = story?.provenance?.find((p) => p.id === state.selectedToken.fact_id);
  const vectors = state.selectedToken.vectors || prov?.vectors || fact?.vectors || [];
  const periods = state.selectedToken.periods || prov?.periods || fact?.periods || [];
  const cut = prov?.cut || fact?.cut || {};
  const heldAt = prov?.held_at || fact?.held_at || {};
  const tableUrl = prov?.table_url || story?.table_url || "";

  details.className = "source-block";
  details.innerHTML = `
    <div>
      <span class="source-row-label">Verified table figure</span>
      <p class="source-claim-quote">${escapeHtml(fact?.human || state.selectedToken.text)}</p>
    </div>
    <div>
      <span class="source-row-label">What is being compared</span>
      ${renderKvChips(cut)}
    </div>
    <div>
      <span class="source-row-label">Population &amp; factors held constant</span>
      ${renderKvChips(heldAt)}
    </div>
    <div>
      <span class="source-row-label">Comparison period</span>
      <div>${escapeHtml(formatPeriodList(periods))}</div>
    </div>
    <div>
      <span class="source-row-label">Statistics Canada series ID(s)</span>
      <div class="kv-chip-list">
        ${vectors.map((v) => `<span class="kv-chip">${escapeHtml(v)}</span>`).join("")}
      </div>
    </div>
    ${
      tableUrl
        ? `<div>
             <a class="link-btn" href="${escapeHtml(tableUrl)}" target="_blank" rel="noopener noreferrer">Open source table on StatCan &#8599;</a>
           </div>`
        : ""
    }
  `;
}

function renderFindings(stories) {
  const listEl = qs("stories-list");
  listEl.innerHTML = "";

  if (!stories || stories.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state">
        <strong>No new findings passed all quality checks for this release.</strong>
        <p class="muted-note" style="margin-top:6px;">
          Check <em>All Checked Items</em> to see which table candidates were already covered in <em>The Daily</em> or filtered out by reliability and comparability safeguards.
        </p>
      </div>
    `;
    renderSourceDrawer();
    return;
  }

  stories.forEach((story) => {
    const card = document.createElement("article");
    card.className = "story-card";

    const rules = findRulesForBrief(story.brief_id);
    const header = document.createElement("div");
    header.className = "story-header";
    header.innerHTML = `
      <span class="badge badge-teal">Ready to share</span>
      <span class="badge badge-navy">${escapeHtml(formatStanceLabel(story.stance))}</span>
      ${
        story.recovered_via_retry
          ? '<span class="badge badge-amber" title="An initial draft needed clearer source grounding and was revised during quality review.">Revised during review</span>'
          : ""
      }
    `;

    const headline = document.createElement("h3");
    headline.className = "story-headline";
    headline.textContent = story.headline || story.title;

    const contextGrid = document.createElement("div");
    contextGrid.className = "story-context-grid";
    contextGrid.innerHTML = `
      <div>
        <span class="context-label">Policy context</span>
        <p class="context-text">${escapeHtml(story.assumption || "Provides additional disaggregation from official survey tables.")}</p>
      </div>
      <div>
        <span class="context-label">What The Daily did not cover</span>
        <p class="context-text">${escapeHtml(story.differs_from_daily || "Extends beyond the headline figures published in the release summary.")}</p>
      </div>
    `;

    const bodyEl = document.createElement("div");
    bodyEl.className = "story-body";
    const paragraphs = splitParagraphsWithSegments(story.body_segments || []);
    paragraphs.forEach((segs) => {
      const p = document.createElement("p");
      segs.forEach((seg) => {
        if (seg.kind === "text") {
          p.appendChild(document.createTextNode(seg.text));
        } else if (seg.kind === "token") {
          const link = document.createElement("a");
          link.href = "#source-drawer";
          link.setAttribute("role", "button");
          link.className = "figure-link";
          if (
            state.selectedStoryId === story.brief_id &&
            state.selectedToken?.fact_id === seg.fact_id
          ) {
            link.classList.add("selected");
          }
          link.textContent = seg.text;
          link.title = "Click to inspect Statistics Canada source details";
          link.addEventListener("click", (evt) => {
            evt.preventDefault();
            state.selectedStoryId = story.brief_id;
            state.selectedToken = seg;
            renderFindings(stories);
            renderSourceDrawer();
          });
          p.appendChild(link);
        }
      });
      bodyEl.appendChild(p);
    });

    const verifyHint = document.createElement("p");
    verifyHint.className = "story-verify-hint";
    verifyHint.textContent =
      "Underlined figures link directly to verified Statistics Canada vectors — click any underlined statement to check its source.";

    card.appendChild(header);
    card.appendChild(headline);
    card.appendChild(contextGrid);
    card.appendChild(bodyEl);
    card.appendChild(verifyHint);

    if (story.critic_note) {
      const note = document.createElement("div");
      note.className = "story-reviewer-note";
      note.innerHTML = `<strong>Quality check note:</strong> ${escapeHtml(formatStatCanReason(story.critic_note))}`;
      card.appendChild(note);
    }

    const actions = document.createElement("div");
    actions.className = "story-actions";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn-secondary";
    copyBtn.textContent = "Copy brief with citation";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(buildCopyText(story));
        copyBtn.textContent = "Copied to clipboard ✓";
        setTimeout(() => {
          copyBtn.textContent = "Copy brief with citation";
        }, 1800);
      } catch (_) {
        copyBtn.textContent = "Select text above to copy";
      }
    });
    actions.appendChild(copyBtn);

    if (story.table_url) {
      const extLink = document.createElement("a");
      extLink.className = "link-btn";
      extLink.href = story.table_url;
      extLink.target = "_blank";
      extLink.rel = "noopener noreferrer";
      extLink.innerHTML = "View StatCan source table &#8599;";
      actions.appendChild(extLink);
    }
    card.appendChild(actions);

    if (rules.length > 0) {
      const details = document.createElement("details");
      details.className = "story-rules-details";
      details.innerHTML = `
        <summary>Methodology rules verified (${rules.length})</summary>
        <ul class="story-rules-list">
          ${rules
            .map(
              (r) =>
                `<li><strong>${escapeHtml(r.title)}:</strong> ${escapeHtml(r.rule)}</li>`
            )
            .join("")}
        </ul>
      `;
      card.appendChild(details);
    }

    listEl.appendChild(card);
  });

  renderSourceDrawer();
}

function renderCoverageManifest(manifest) {
  const box = qs("coverage-manifest-box");
  if (!manifest || !Array.isArray(manifest.claimed_coordinates) || manifest.claimed_coordinates.length === 0) {
    box.innerHTML = `
      <h3>Already Covered by Statistics Canada</h3>
      <p class="section-intro" style="margin-bottom:0;">
        This release checks candidate items directly against headlines mentioned in <em>The Daily</em> so analysts are not shown duplicate findings.
      </p>
    `;
    return;
  }

  const coords = manifest.claimed_coordinates;
  box.innerHTML = `
    <h3>Already Covered in Official Reports (${coords.length} reference points tracked)</h3>
    <p class="section-intro">
      REMINE extracts claims already reported in <em>The Daily</em>${manifest.paper_catalogue ? ` and <em>${escapeHtml(manifest.paper_catalogue)}</em>` : ""} to avoid surfacing numbers your team has already read.
    </p>
    <div class="coverage-grid">
      ${coords
        .map((c) => {
          const tierLabel = c.tier === "paper" ? "Analytical report" : "The Daily bulletin";
          const cls = c.tier === "paper" ? "badge-navy" : "badge-muted";
          const dimsText = formatKvPlainText(c.dimensions);
          return `
            <div class="coverage-item">
              <div class="ledger-meta-row">
                <span class="badge ${cls}">${escapeHtml(tierLabel)}</span>
                ${c.period ? `<span class="badge badge-muted">${escapeHtml(c.period)}</span>` : ""}
              </div>
              <div style="font-size:0.86rem;font-weight:600;color:var(--ink);">
                ${escapeHtml(dimsText || "Headline estimate")}
              </div>
              <p class="coverage-quote">“${escapeHtml(
                String(c.prose_quote || "").replace(
                  /^The Daily\s*\|.*?Release schedule Information\s*/i,
                  ""
                )
              )}”</p>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderLedger(ledger) {
  const listEl = qs("ledger-list");
  listEl.innerHTML = "";
  if (!ledger || ledger.length === 0) {
    listEl.innerHTML = `<div class="empty-state">No table items checked in this release.</div>`;
    return;
  }

  ledger.forEach((row) => {
    const vMeta = formatVerdictMeta(row.verdict, row.rejection_category);
    const card = document.createElement("div");
    card.className = "ledger-card";
    card.innerHTML = `
      <div class="ledger-top">
        <h4 class="ledger-title">${escapeHtml(row.title)}</h4>
        <span class="badge ${vMeta.cls}">${escapeHtml(vMeta.label)}</span>
      </div>
      <p class="ledger-summary">${escapeHtml(formatStatCanReason(row.summary || ""))}</p>
      <div class="ledger-meta-row">
        <span class="badge badge-muted">${escapeHtml(formatQualityFlag(row.quality_flag))}</span>
        <span class="badge ${row.standardized === false ? "badge-amber" : "badge-muted"}">
          ${row.standardized === false ? "Unadjusted subgroup mix" : "Comparable subgroups"}
        </span>
      </div>
      ${
        row.reason
          ? `<p class="ledger-reason"><strong>Screening decision:</strong> ${escapeHtml(formatStatCanReason(row.reason))}</p>`
          : ""
      }
      ${
        row.critic_note
          ? `<p class="ledger-reason"><strong>Quality reviewer note:</strong> ${escapeHtml(formatStatCanReason(row.critic_note))}</p>`
          : ""
      }
    `;
    listEl.appendChild(card);
  });
}

function renderReviewQueue(queue) {
  const listEl = qs("review-queue-list");
  listEl.innerHTML = "";
  if (!queue || queue.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state">
        No items were held back for data suppression or analyst escalation in this release.
      </div>
    `;
    return;
  }

  queue.forEach((row) => {
    const vMeta = formatVerdictMeta(row.verdict, row.rejection_category);
    const card = document.createElement("div");
    card.className = "ledger-card";
    card.innerHTML = `
      <div class="ledger-top">
        <h4 class="ledger-title">${escapeHtml(row.title)}</h4>
        <span class="badge ${vMeta.cls}">${escapeHtml(vMeta.label)}</span>
      </div>
      <p class="ledger-summary">${escapeHtml(formatStatCanReason(row.summary || ""))}</p>
      <div class="ledger-meta-row">
        <span class="badge badge-crimson">${escapeHtml(formatQualityFlag(row.quality_flag))}</span>
      </div>
      <p class="ledger-reason"><strong>Why it was held back:</strong> ${escapeHtml(formatStatCanReason(row.reason))}</p>
    `;
    listEl.appendChild(card);
  });
}

function renderProcessAndBenchmarks(data) {
  qs("process-steps").innerHTML = PROCESS_STEPS.map(
    (step) => `
      <li class="process-step">
        <span class="step-num">${step.num}</span>
        <h4>${escapeHtml(step.title)}</h4>
        <p>${escapeHtml(step.desc)}</p>
      </li>
    `
  ).join("");

  const runs = state.benchmarks?.runs || [];
  const wrap = qs("benchmarks-table-wrap");
  if (runs.length === 0) {
    wrap.innerHTML = `<p class="muted-note">No cross-model screening benchmarks recorded yet.</p>`;
  } else {
    wrap.innerHTML = `
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Release Dataset</th>
              <th>Automated Reviewer</th>
              <th>Comparability Guard</th>
              <th>Correct Decisions</th>
              <th>Drafts Revised &amp; Fixed</th>
              <th>Time Taken</th>
            </tr>
          </thead>
          <tbody>
            ${runs
              .map((r) => {
                const accPct =
                  typeof r.verdict_accuracy === "number"
                    ? `${Math.round(r.verdict_accuracy * 100)}% (${r.correct_verdicts}/${r.total_briefs})`
                    : "—";
                return `
                  <tr>
                    <td>${escapeHtml(r.dataset)}</td>
                    <td>${escapeHtml(formatModelLabel(r.model))}</td>
                    <td>${r.gate0 ? "On (strict)" : "Off (caught later)"}</td>
                    <td>${escapeHtml(accPct)}</td>
                    <td>${escapeHtml(r.recovered_via_retry ?? 0)}</td>
                    <td>${escapeHtml(r.elapsed_s != null ? `${r.elapsed_s}s` : "—")}</td>
                  </tr>
                `;
              })
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  const tracePanel = qs("technical-trace-panel");
  const ledger = data.ledger || [];
  tracePanel.innerHTML = `
    <div class="kv-chip-list">
      <span class="kv-chip"><strong>Model engine:</strong> ${escapeHtml(data.config?.model || "replay")}</span>
      <span class="kv-chip"><strong>Comparable-subgroup guard:</strong> ${data.config?.enforce_semantic_standardization ? "Enabled" : "Disabled"}</span>
      <span class="kv-chip"><strong>Model calls:</strong> ${escapeHtml(data.summary?.llm_calls ?? 0)}</span>
      <span class="kv-chip"><strong>Tool lookups:</strong> ${escapeHtml(data.summary?.mcp_calls ?? 0)}</span>
    </div>
    ${ledger
      .map(
        (row) => `
          <div class="coverage-item">
            <strong>${escapeHtml(row.brief_id)} — ${escapeHtml(row.title)}</strong>
            <ul style="margin:6px 0 0;padding-left:18px;font-size:0.84rem;color:var(--ink-muted);">
              ${(row.trace || [])
                .map(
                  (ev) =>
                    `<li><code>${escapeHtml(ev.node)}</code> (${escapeHtml(ev.status)}) — ${escapeHtml(ev.summary)}</li>`
                )
                .join("")}
            </ul>
          </div>
        `
      )
      .join("")}
  `;
}

function applyRunData(data) {
  state.activeRun = data;
  state.selectedToken = null;
  state.selectedStoryId = null;

  if (Array.isArray(data.stories) && data.stories.length > 0) {
    const firstSeg = (data.stories[0].body_segments || []).find((s) => s.kind === "token");
    if (firstSeg) {
      state.selectedStoryId = data.stories[0].brief_id;
      state.selectedToken = firstSeg;
    }
  }

  renderReleaseAndSummary(data);
  renderFindings(data.stories || []);
  renderCoverageManifest(data.coverage_manifest);
  renderLedger(data.ledger || []);
  renderReviewQueue(data.review_queue || []);
  renderProcessAndBenchmarks(data);
}

async function runActiveSelection() {
  const snapshotId = qs("snapshot-select").value;
  const strictComparability = qs("gate0-toggle").checked;
  if (!snapshotId) return;

  showError("");
  state.highlightsExpanded = false;

  if (state.runtimeMode === "static") {
    const runPair = state.staticBundle?.runs?.[snapshotId];
    const picked = runPair?.[strictComparability ? "strict" : "off"];
    if (!picked) {
      showError("Pre-verified screening data is not available for this release.");
      return;
    }
    applyRunData(picked);
    return;
  }

  const runBtn = qs("run-btn");
  runBtn.disabled = true;
  runBtn.textContent = "Reviewing…";
  try {
    const payload = {
      snapshot_id: snapshotId,
      mode: qs("mode-select").value,
      model: qs("model-select").value,
      enforce_semantic_standardization: strictComparability,
      use_mcp: true,
    };
    const res = await fetch("api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Unable to complete release review.");
    }
    applyRunData(data);
  } catch (err) {
    const msg = err.message || String(err);
    if (msg.includes("no replay response")) {
      showError(
        "This freshly fetched release does not have a saved pre-verified script yet. Open 'Local server tools' above and choose 'Local AI assistant' to review it."
      );
    } else {
      showError(msg);
    }
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Update Review";
  }
}

async function harvestRelease() {
  const btn = qs("harvest-btn");
  const feedback = qs("harvest-feedback");
  btn.disabled = true;
  btn.textContent = "Fetching…";
  feedback.hidden = false;
  feedback.textContent = "Downloading The Daily summary and Statistics Canada table metadata…";
  showError("");
  try {
    const res = await fetch("api/harvest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        date: qs("harvest-date").value.trim() || null,
        pid: qs("harvest-pid").value.trim() || null,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Fetch failed");
    }
    state.snapshots = data.snapshots || [];
    populateSelect(
      qs("snapshot-select"),
      state.snapshots,
      (s) => s.id,
      (s) => `${s.title} (${formatDateLabel(s.date) || s.id})`
    );
    if (data.snapshot_id) {
      qs("snapshot-select").value = data.snapshot_id;
    }
    feedback.textContent = `Saved "${data.harvested?.article || data.snapshot_id}" with ${data.harvested?.brief_count ?? 0} candidate table items.`;
  } catch (err) {
    feedback.hidden = true;
    showError(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Fetch Release";
  }
}

function wireEvents() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.remove("active");
        pane.hidden = true;
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const target = qs(`tab-${btn.dataset.tab}`);
      if (target) {
        target.classList.add("active");
        target.hidden = false;
      }
    });
  });

  qs("snapshot-select").addEventListener("change", () => {
    if (state.runtimeMode === "static") {
      runActiveSelection();
    } else {
      const selected = state.snapshots.find((s) => s.id === qs("snapshot-select").value);
      if (selected?.source === "harvested" && state.health?.local_llm?.online) {
        qs("mode-select").value = "live";
      } else {
        qs("mode-select").value = "replay";
      }
    }
  });

  qs("gate0-toggle").addEventListener("change", () => {
    runActiveSelection();
  });

  qs("run-btn").addEventListener("click", runActiveSelection);
  qs("harvest-btn").addEventListener("click", harvestRelease);
  qs("toggle-highlights-btn").addEventListener("click", () => {
    state.highlightsExpanded = !state.highlightsExpanded;
    if (state.activeRun) renderReleaseAndSummary(state.activeRun);
  });
}

async function init() {
  wireEvents();

  // Detect live Remine HTTP API vs static GitHub Pages bundle
  try {
    const healthRes = await fetch("api/health", { cache: "no-store" });
    const ct = healthRes.headers.get("content-type") || "";
    if (healthRes.ok && ct.includes("application/json")) {
      state.runtimeMode = "server";
      state.health = await healthRes.json();
    } else {
      state.runtimeMode = "static";
    }
  } catch (_) {
    state.runtimeMode = "static";
  }

  setStatusBadge();
  setServerControlsVisibility();

  try {
    if (state.runtimeMode === "server") {
      const [snapsRes, benchRes] = await Promise.all([
        fetch("api/snapshots", { cache: "no-store" }),
        fetch("api/benchmarks", { cache: "no-store" }),
      ]);
      const snapsData = await snapsRes.json();
      state.benchmarks = await benchRes.json();
      state.snapshots = snapsData.snapshots || [];

      const llmModels = state.health?.local_llm?.models || [];
      const modelOptions = llmModels.length > 0 ? llmModels : [state.health?.default_model || "qwen3.8-27b"];
      populateSelect(qs("model-select"), modelOptions, (m) => m, (m) => m);
    } else {
      const staticRes = await fetch("static/site_data.json", { cache: "no-store" });
      if (!staticRes.ok) {
        throw new Error("Could not load pre-verified Statistics Canada archive data.");
      }
      state.staticBundle = await staticRes.json();
      state.snapshots = state.staticBundle.snapshots || [];
      state.benchmarks = state.staticBundle.benchmarks || { runs: [] };
    }

    populateSelect(
      qs("snapshot-select"),
      state.snapshots,
      (s) => s.id,
      (s) => `${s.title} (${formatDateLabel(s.date) || s.id})`
    );

    // Default to Labour Force Survey benchmark if present
    const defaultSnap =
      state.snapshots.find((s) => s.id === "lfs_260807_snapshot") || state.snapshots[0];
    if (defaultSnap) {
      qs("snapshot-select").value = defaultSnap.id;
    }

    await runActiveSelection();
  } catch (err) {
    showError(`Unable to initialize viewer: ${err.message || err}`);
  }
}

document.addEventListener("DOMContentLoaded", init);
