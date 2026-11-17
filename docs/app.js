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
  wfBriefId: null,
  wfStepIndex: 0,
  wfTimer: null,
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
    desc: "A second check confirms all qualifiers, time periods, and comparison caveats are stated accurately, or sends the draft back for revision.",
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
    return { label: "Ready to share", cls: "verdict-teal" };
  }
  if (verdict === "unreliable_data") {
    return { label: "Held back: unreliable or suppressed", cls: "verdict-crimson" };
  }
  if (verdict === "needs_editorial_review") {
    return { label: "Held back: needs analyst review", cls: "verdict-amber" };
  }
  if (verdict === "skip") {
    const reasonMap = {
      ALREADY_IN_THE_DAILY: "Filtered: already in The Daily",
      ALREADY_IN_ANALYTICAL_PAPER: "Filtered: covered in StatCan report",
      CONFOUNDED_COMPARISON: "Filtered: skewed subgroup comparison",
    };
    return {
      label: reasonMap[rejectionCategory] || "Filtered out during screening",
      cls: rejectionCategory === "CONFOUNDED_COMPARISON" ? "verdict-amber" : "verdict-muted",
    };
  }
  return { label: verdict || "Screened", cls: "verdict-muted" };
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
    pill.textContent = "Archive edition";
    pill.className = "status-text";
    return;
  }
  const aiOnline = Boolean(state.health?.local_llm?.online);
  pill.textContent = aiOnline ? "Local server (AI ready)" : "Local server";
  pill.className = "status-text online";
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
  } else {
    qs("kpi-accuracy").textContent = `${total} checked`;
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
    <p class="source-claim-quote">“${escapeHtml(fact?.human || state.selectedToken.text)}”</p>
    <table class="source-kv-table">
      <tbody>
        <tr>
          <th>Compared</th>
          <td>${escapeHtml(formatKvPlainText(cut) || "None")}</td>
        </tr>
        <tr>
          <th>Held constant</th>
          <td>${escapeHtml(formatKvPlainText(heldAt) || "None")}</td>
        </tr>
        <tr>
          <th>Period</th>
          <td>${escapeHtml(formatPeriodList(periods))}</td>
        </tr>
        <tr>
          <th>StatCan series</th>
          <td>${escapeHtml(vectors.join(", ") || "Not specified")}</td>
        </tr>
      </tbody>
    </table>
    ${
      tableUrl
        ? `<a class="text-link" href="${escapeHtml(tableUrl)}" target="_blank" rel="noopener noreferrer">Open source table on StatCan &#8599;</a>`
        : ""
    }
  `;
}

function buildStoryBreakdownTables(facts) {
  const groups = [];
  (facts || []).forEach((f) => {
    const bd = f.breakdown;
    if (!bd || !Array.isArray(bd.columns) || !Array.isArray(bd.rows) || bd.rows.length === 0) return;
    const key = bd.columns.join("||");
    const existing = groups.find((g) => g.key === key);
    if (existing) {
      existing.rows.push(...bd.rows);
    } else {
      groups.push({ key, columns: bd.columns, rows: [...bd.rows] });
    }
  });
  return groups.map(
    (g) => `
      <div class="story-figures-wrap">
        <div class="table-wrap">
          <table class="data-table compact">
            <thead>
              <tr>${g.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>
            </thead>
            <tbody>
              ${g.rows
                .map(
                  (row) =>
                    `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `
  );
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
    const entry = document.createElement("article");
    entry.className = "story-entry";

    const rules = findRulesForBrief(story.brief_id);
    const kicker = document.createElement("div");
    kicker.className = "story-kicker";
    kicker.innerHTML = `
      Ready to share &middot; ${escapeHtml(formatStanceLabel(story.stance))}
      ${
        story.recovered_via_retry
          ? ' &middot; <span class="revision-note">Revised during quality check</span>'
          : ""
      }
    `;

    const headline = document.createElement("h3");
    headline.className = "story-headline";
    headline.textContent = story.headline || story.title;

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

    const notes = document.createElement("div");
    notes.className = "story-notes";
    const noteLines = [];
    if (story.differs_from_daily) {
      noteLines.push(`<div><strong>Not in The Daily:</strong> ${escapeHtml(story.differs_from_daily)}</div>`);
    }
    if (story.assumption) {
      noteLines.push(`<div><strong>Context:</strong> ${escapeHtml(story.assumption)}</div>`);
    }
    if (story.critic_note) {
      noteLines.push(`<div><strong>Quality check:</strong> ${escapeHtml(formatStatCanReason(story.critic_note))}</div>`);
    }
    notes.innerHTML = noteLines.join("");

    entry.appendChild(kicker);
    entry.appendChild(headline);
    entry.appendChild(bodyEl);

    const breakdownHtml = buildStoryBreakdownTables(story.facts);
    if (breakdownHtml.length > 0) {
      const figBox = document.createElement("div");
      figBox.className = "story-figures-section";
      figBox.innerHTML = `<div class="story-figures-label">Supporting figures</div>${breakdownHtml.join("")}`;
      entry.appendChild(figBox);
    }

    if (noteLines.length > 0) entry.appendChild(notes);

    const actions = document.createElement("div");
    actions.className = "story-actions";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn-secondary";
    copyBtn.textContent = "Copy brief with citation";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(buildCopyText(story));
        copyBtn.textContent = "Copied";
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
      extLink.className = "text-link";
      extLink.href = story.table_url;
      extLink.target = "_blank";
      extLink.rel = "noopener noreferrer";
      extLink.innerHTML = "View StatCan source table &#8599;";
      actions.appendChild(extLink);
    }
    entry.appendChild(actions);

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
      entry.appendChild(details);
    }

    listEl.appendChild(entry);
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
    <h3>Already Covered in Official Reports (${coords.length} reference points)</h3>
    <p class="section-intro">
      Claims extracted from <em>The Daily</em>${manifest.paper_catalogue ? ` and <em>${escapeHtml(manifest.paper_catalogue)}</em>` : ""} to avoid repeating headline figures.
    </p>
    <div class="coverage-list">
      ${coords
        .map((c) => {
          const tierLabel = c.tier === "paper" ? "Analytical report" : "The Daily";
          const dimsText = formatKvPlainText(c.dimensions);
          return `
            <div>
              <div class="coverage-entry-meta">
                ${escapeHtml(tierLabel)}${c.period ? ` &middot; ${escapeHtml(c.period)}` : ""}
              </div>
              <div class="coverage-entry-dims">
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
    const item = document.createElement("div");
    item.className = "ledger-row";
    item.innerHTML = `
      <div class="ledger-top">
        <h4 class="ledger-title">${escapeHtml(row.title)}</h4>
        <span class="verdict-text ${vMeta.cls}">${escapeHtml(vMeta.label)}</span>
      </div>
      <p class="ledger-summary">${escapeHtml(formatStatCanReason(row.summary || ""))}</p>
      <div class="ledger-meta-line">
        ${escapeHtml(formatQualityFlag(row.quality_flag))} &middot; ${
          row.standardized === false ? "Unadjusted subgroup mix" : "Comparable subgroups"
        }
      </div>
      ${
        row.reason
          ? `<p class="ledger-reason"><strong>Decision:</strong> ${escapeHtml(formatStatCanReason(row.reason))}</p>`
          : ""
      }
      ${
        row.critic_note
          ? `<p class="ledger-reason"><strong>Quality check:</strong> ${escapeHtml(formatStatCanReason(row.critic_note))}</p>`
          : ""
      }
    `;
    listEl.appendChild(item);
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
    const item = document.createElement("div");
    item.className = "ledger-row";
    item.innerHTML = `
      <div class="ledger-top">
        <h4 class="ledger-title">${escapeHtml(row.title)}</h4>
        <span class="verdict-text ${vMeta.cls}">${escapeHtml(vMeta.label)}</span>
      </div>
      <p class="ledger-summary">${escapeHtml(formatStatCanReason(row.summary || ""))}</p>
      <div class="ledger-meta-line">${escapeHtml(formatQualityFlag(row.quality_flag))}</div>
      <p class="ledger-reason"><strong>Reason held back:</strong> ${escapeHtml(formatStatCanReason(row.reason))}</p>
    `;
    listEl.appendChild(item);
  });
}

function renderProcessAndBenchmarks(data) {
  qs("process-steps").innerHTML = PROCESS_STEPS.map(
    (step) => `
      <li class="process-item">
        <span class="process-num">${step.num}.</span>
        <span class="process-title">${escapeHtml(step.title)}</span>
        <p class="process-desc">${escapeHtml(step.desc)}</p>
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

const WF_NODE_META = {
  gate: { label: "Gate 0", sub: "Quality & Guard", x: 16, y: 42, w: 104, h: 54 },
  retrieve: { label: "MCP Lookup", sub: "Rules & Manifest", x: 144, y: 42, w: 106, h: 54 },
  angle: { label: "Angle Check", sub: "Novelty Screen", x: 274, y: 42, w: 108, h: 54 },
  draft: { label: "Drafter", sub: "Locked Figures", x: 406, y: 42, w: 108, h: 54 },
  bind_check: { label: "Token Check", sub: "Numeral Contract", x: 538, y: 42, w: 108, h: 54 },
  critic: { label: "Critic", sub: "Methods Audit", x: 670, y: 42, w: 104, h: 54 },
  published: { label: "Verified", sub: "Ready to Share", x: 798, y: 42, w: 106, h: 54 },
  filtered: { label: "Filtered / Held", sub: "Skipped or Suppressed", x: 140, y: 184, w: 176, h: 50 },
  review: { label: "Human Review", sub: "Escalated Safety", x: 608, y: 184, w: 170, h: 50 },
};

function stopWorkflowPlayback() {
  if (state.wfTimer) {
    clearInterval(state.wfTimer);
    state.wfTimer = null;
  }
  const btn = qs("wf-play-btn");
  if (btn) btn.textContent = "Play";
}

function buildWorkflowSteps(row) {
  const trace = Array.isArray(row?.trace) ? row.trace : [];
  const steps = trace.map((ev, idx) => ({
    idx,
    node: ev.node,
    status: ev.status,
    attempt: ev.attempt || 0,
    summary: formatStatCanReason(ev.summary || ""),
    isTerminal: false,
  }));
  if (row) {
    const v = row.verdict;
    let outcomeNode = "published";
    if (v === "skip" || v === "unreliable_data") outcomeNode = "filtered";
    else if (v === "needs_editorial_review") outcomeNode = "review";
    const lastEv = trace[trace.length - 1];
    steps.push({
      idx: steps.length,
      node: outcomeNode,
      status: v,
      attempt: row.attempts || lastEv?.attempt || 0,
      summary: formatStatCanReason(
        row.reason || row.critic_note || formatVerdictMeta(v, row.rejection_category).label
      ),
      isTerminal: true,
    });
  }
  return steps;
}

function renderWorkflowFrame() {
  const row = (state.activeRun?.ledger || []).find((r) => r.brief_id === state.wfBriefId);
  const svgWrap = qs("wf-svg-wrap");
  const pillsEl = qs("wf-step-pills");
  const logEl = qs("wf-step-log");
  if (!row) {
    svgWrap.innerHTML = "";
    pillsEl.innerHTML = "";
    logEl.innerHTML = "No items available to inspect.";
    return;
  }

  const steps = buildWorkflowSteps(row);
  const maxIdx = Math.max(0, steps.length - 1);
  if (state.wfStepIndex > maxIdx) state.wfStepIndex = maxIdx;
  const activeStep = steps[state.wfStepIndex];
  const visibleSteps = steps.slice(0, state.wfStepIndex + 1);

  const nodeState = {};
  const edgeState = {};
  visibleSteps.forEach((st, i) => {
    const isCurrent = i === state.wfStepIndex;
    let kind = "visited";
    if (st.status === "retry" || st.status === "rejected") kind = "retry";
    else if (st.status === "blocked" || st.status === "skip" || st.node === "filtered" || st.node === "review") {
      kind = "blocked";
    } else if (isCurrent) {
      kind = "active";
    }
    nodeState[st.node] = kind;
    if (i > 0) {
      const prevNode = visibleSteps[i - 1].node;
      const eKey = `${prevNode}->${st.node}`;
      edgeState[eKey] = {
        cls:
          kind === "retry"
            ? "retry-traversed"
            : kind === "blocked"
              ? "blocked-traversed"
              : "traversed",
        active: isCurrent,
      };
    }
  });

  const edges = [
    { key: "gate->retrieve", d: "M 120 69 L 144 69" },
    { key: "retrieve->angle", d: "M 250 69 L 274 69" },
    { key: "angle->draft", d: "M 382 69 L 406 69" },
    { key: "draft->bind_check", d: "M 514 69 L 538 69" },
    { key: "bind_check->critic", d: "M 646 69 L 670 69" },
    { key: "critic->published", d: "M 774 69 L 798 69" },
    { key: "gate->filtered", d: "M 68 96 L 68 209 L 140 209" },
    { key: "angle->filtered", d: "M 328 96 L 328 150 L 245 150 L 245 184" },
    { key: "bind_check->draft", d: "M 592 96 L 592 136 L 460 136 L 460 96" },
    { key: "critic->draft", d: "M 722 96 L 722 148 L 460 148 L 460 96" },
    { key: "bind_check->review", d: "M 592 96 L 592 209 L 608 209" },
    { key: "critic->review", d: "M 722 96 L 722 184" },
    { key: "angle->review", d: "M 328 96 L 328 164 L 660 164 L 660 184" },
  ];

  const edgeSvg = edges
    .map((e) => {
      const st = edgeState[e.key];
      const cls = ["wf-edge", st ? st.cls : "", st?.active ? "active" : ""].filter(Boolean).join(" ");
      const marker = st?.active || st ? "url(#wf-arrow-active)" : "url(#wf-arrow)";
      return `<path class="${cls}" d="${e.d}" marker-end="${marker}" />`;
    })
    .join("");

  const nodeSvg = Object.entries(WF_NODE_META)
    .map(([id, m]) => {
      const st = nodeState[id] || "idle";
      return `
        <g class="wf-node ${st}">
          <rect x="${m.x}" y="${m.y}" width="${m.w}" height="${m.h}" rx="3" />
          <text x="${m.x + m.w / 2}" y="${m.y + 22}" text-anchor="middle" font-size="12" font-weight="600">${escapeHtml(m.label)}</text>
          <text x="${m.x + m.w / 2}" y="${m.y + 39}" text-anchor="middle" font-size="10">${escapeHtml(m.sub)}</text>
        </g>
      `;
    })
    .join("");

  svgWrap.innerHTML = `
    <svg class="wf-svg" viewBox="0 0 920 250" role="img" aria-label="Screening workflow state diagram">
      <defs>
        <marker id="wf-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#b8b3a6" />
        </marker>
        <marker id="wf-arrow-active" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1b4965" />
        </marker>
      </defs>
      ${edgeSvg}
      <text x="530" y="131" text-anchor="middle" font-size="10" fill="#6b7280">Retry loop with feedback</text>
      ${nodeSvg}
    </svg>
  `;

  pillsEl.innerHTML = steps
    .map((st, i) => {
      const nodeLabel = WF_NODE_META[st.node]?.label || st.node;
      let cls = "wf-step-btn";
      if (i === state.wfStepIndex) cls += " current";
      else if (st.status === "retry" || st.status === "rejected") cls += " retry";
      else if (st.status === "blocked" || st.status === "skip") cls += " blocked";
      else if (i < state.wfStepIndex) cls += " done";
      const suffix = st.attempt > 0 && st.node === "draft" ? ` (#${st.attempt})` : "";
      return `<button type="button" class="${cls}" data-wf-step="${i}">${i + 1}. ${escapeHtml(nodeLabel)}${suffix}</button>`;
    })
    .join("");

  pillsEl.querySelectorAll("[data-wf-step]").forEach((btn) => {
    btn.addEventListener("click", () => {
      stopWorkflowPlayback();
      state.wfStepIndex = Number(btn.dataset.wfStep);
      renderWorkflowFrame();
    });
  });

  if (activeStep) {
    const label = WF_NODE_META[activeStep.node]?.label || activeStep.node;
    logEl.innerHTML = `
      <div class="wf-step-log-head">
        <span>Step ${state.wfStepIndex + 1} of ${steps.length}: ${escapeHtml(label)}${activeStep.attempt ? ` (Attempt ${activeStep.attempt})` : ""}</span>
        <span>${escapeHtml(activeStep.isTerminal ? "Final outcome" : `Status: ${activeStep.status}`)}</span>
      </div>
      <div>${escapeHtml(activeStep.summary)}</div>
    `;
  }
}

function renderWorkflowExplorer(data) {
  stopWorkflowPlayback();
  const ledger = data.ledger || [];
  const select = qs("wf-brief-select");
  populateSelect(
    select,
    ledger,
    (r) => r.brief_id,
    (r) => `${r.brief_id}: ${String(r.title || "").slice(0, 48)}`
  );
  if (ledger.length > 0) {
    const retryRow = ledger.find((r) => (r.trace || []).some((ev) => ev.status === "retry"));
    const picked = ledger.find((r) => r.brief_id === state.wfBriefId) || retryRow || ledger[0];
    state.wfBriefId = picked.brief_id;
    select.value = picked.brief_id;
    const steps = buildWorkflowSteps(picked);
    state.wfStepIndex = Math.max(0, steps.length - 1);
  } else {
    state.wfBriefId = null;
    state.wfStepIndex = 0;
  }
  renderWorkflowFrame();
}

function startWorkflowPlayback() {
  if (state.wfTimer) {
    stopWorkflowPlayback();
    return;
  }
  const row = (state.activeRun?.ledger || []).find((r) => r.brief_id === state.wfBriefId);
  const steps = buildWorkflowSteps(row);
  if (steps.length === 0) return;
  if (state.wfStepIndex >= steps.length - 1) {
    state.wfStepIndex = 0;
    renderWorkflowFrame();
  }
  qs("wf-play-btn").textContent = "Pause";
  state.wfTimer = setInterval(() => {
    if (state.wfStepIndex >= steps.length - 1) {
      stopWorkflowPlayback();
      return;
    }
    state.wfStepIndex += 1;
    renderWorkflowFrame();
  }, 900);
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
  renderWorkflowExplorer(data);
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

  qs("wf-brief-select").addEventListener("change", () => {
    stopWorkflowPlayback();
    state.wfBriefId = qs("wf-brief-select").value;
    state.wfStepIndex = 0;
    renderWorkflowFrame();
    startWorkflowPlayback();
  });

  qs("wf-play-btn").addEventListener("click", startWorkflowPlayback);

  qs("wf-step-btn").addEventListener("click", () => {
    stopWorkflowPlayback();
    const row = (state.activeRun?.ledger || []).find((r) => r.brief_id === state.wfBriefId);
    const steps = buildWorkflowSteps(row);
    if (steps.length === 0) return;
    state.wfStepIndex = state.wfStepIndex < steps.length - 1 ? state.wfStepIndex + 1 : 0;
    renderWorkflowFrame();
  });

  qs("wf-reset-btn").addEventListener("click", () => {
    stopWorkflowPlayback();
    state.wfStepIndex = 0;
    renderWorkflowFrame();
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
