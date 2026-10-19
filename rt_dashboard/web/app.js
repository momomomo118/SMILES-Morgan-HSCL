const $ = (id) => document.getElementById(id);

const state = { examples: [], busy: false, result: null };

function setBusy(busy) {
  state.busy = busy;
  $("predictButton").disabled = busy;
  $("loadingBar").hidden = !busy;
  $("predictButton").querySelector(".button-label").textContent = busy ? "正在预测" : "开始预测";
}

async function getJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}

async function initialize() {
  const [metrics, examples, status] = await Promise.all([
    getJSON("/api/metrics"), getJSON("/api/examples"), getJSON("/api/status")
  ]);
  $("maeMetric").innerHTML = `${metrics.mae.toFixed(3)} <small>s</small>`;
  $("r2Metric").textContent = metrics.r2.toFixed(4);
  $("rmseMetric").innerHTML = `${metrics.rmse.toFixed(3)} <small>s</small>`;
  state.examples = examples.examples;
  renderExamples();
  updateStatus(status);
  drawBranchChart(null);
}

function renderExamples() {
  const root = $("exampleChips");
  root.innerHTML = "";
  state.examples.forEach((example) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.textContent = example.name;
    button.title = example.smiles;
    button.addEventListener("click", () => {
      $("smilesInput").value = example.smiles;
      updateCharCount();
      $("smilesInput").focus();
    });
    root.appendChild(button);
  });
}

function updateStatus(status) {
  const pill = $("healthPill");
  $("deviceValue").textContent = status.device;
  if (status.error) {
    pill.textContent = "异常";
    pill.className = "health-pill error";
  } else if (status.loaded) {
    pill.textContent = "就绪";
    pill.className = "health-pill";
  } else if (status.loading) {
    pill.textContent = "加载中";
    pill.className = "health-pill loading";
  } else {
    pill.textContent = "待加载";
    pill.className = "health-pill loading";
  }
}

function updateCharCount() {
  const length = $("smilesInput").value.replace(/\s/g, "").length;
  $("charCount").textContent = `${length} / 512`;
  $("charCount").style.color = length > 512 ? "#c85f75" : "";
}

async function predict() {
  if (state.busy) return;
  const smiles = $("smilesInput").value.trim();
  $("errorBox").hidden = true;
  if (!smiles) {
    showError("请输入 SMILES 序列后再开始预测。");
    return;
  }
  setBusy(true);
  try {
    const payload = await getJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles })
    });
    state.result = payload.result;
    renderResult(payload.result);
    updateStatus(await getJSON("/api/status"));
  } catch (error) {
    showError(error.message);
    updateStatus(await getJSON("/api/status").catch(() => ({ error: true, device: "—" })));
  } finally {
    setBusy(false);
  }
}

function showError(message) {
  $("errorBox").textContent = message;
  $("errorBox").hidden = false;
}

function renderResult(result) {
  $("resultContent").hidden = false;
  $("resultPanel").classList.remove("empty-state");
  $("resultTimestamp").textContent = new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  $("predictionValue").textContent = result.prediction_s.toFixed(2);
  $("predictionMinutes").textContent = result.prediction_min.toFixed(2);
  $("errorValue").textContent = result.estimated_abs_error_s.toFixed(2);
  $("intervalValue").textContent = `90% 参考区间：${result.interval_90_s[0].toFixed(1)} – ${result.interval_90_s[1].toFixed(1)} s`;
  $("canonicalValue").textContent = result.canonical_smiles;
  $("similarityValue").textContent = `${Math.round(result.nearest_similarity * 100)}%`;
  $("similarityBar").style.width = `${Math.max(3, result.nearest_similarity * 100)}%`;
  $("disagreementValue").textContent = `${result.model_disagreement_s.toFixed(1)} s`;
  $("resultNote").textContent = result.note;
  $("latencyValue").textContent = `${result.latency_s.toFixed(2)} s`;

  const confidenceScore = result.confidence === "高" ? 88 : result.confidence === "中" ? 68 : 43;
  $("confidenceValue").textContent = result.confidence;
  $("confidenceRing").style.setProperty("--value", confidenceScore);

  $("moleculePreview").innerHTML = `<img alt="${escapeHtml(result.molecule.formula)} 分子二维结构" src="${result.molecule.image}">`;
  $("formulaValue").textContent = result.molecule.formula;
  $("weightValue").textContent = `${result.molecule.molecular_weight.toFixed(2)} Da`;
  $("logpValue").textContent = result.molecule.logp.toFixed(2);
  $("tpsaValue").textContent = `${result.molecule.tpsa.toFixed(1)} Å²`;
  $("ringsValue").textContent = result.molecule.rings;
  $("atomsValue").textContent = result.molecule.heavy_atoms;
  drawBranchChart(result.components);
  $("resultPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function drawBranchChart(components) {
  const canvas = $("branchChart");
  const rect = canvas.getBoundingClientRect();
  const scale = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.max(600, rect.width * scale);
  canvas.height = 210 * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  const width = canvas.width / scale;
  const height = canvas.height / scale;
  ctx.clearRect(0, 0, width, height);
  const names = ["CCRL ensemble", "GIN 2026", "GIN 3407", "GIN 777"];
  const values = components ? [components.pred_ccrl_ensemble, components.gnn2026, components.gnn3407, components.gnn777] : [68, 48, 57, 52];
  const max = Math.max(...values) * 1.15 || 1;
  const padding = { left: 42, right: 18, top: 18, bottom: 38 };
  const chartH = height - padding.top - padding.bottom;
  const chartW = width - padding.left - padding.right;

  ctx.font = "9px Inter, sans-serif";
  ctx.fillStyle = "#a2a3b5";
  ctx.strokeStyle = "#eeeeF5";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + chartH * i / 4;
    ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(width - padding.right, y); ctx.stroke();
    const label = Math.round(max * (1 - i / 4));
    ctx.fillText(components ? `${label}s` : "", 5, y + 3);
  }

  const slot = chartW / values.length;
  values.forEach((value, index) => {
    const barW = Math.min(55, slot * .42);
    const x = padding.left + slot * index + (slot - barW) / 2;
    const barH = components ? chartH * value / max : value;
    const y = padding.top + chartH - barH;
    const gradient = ctx.createLinearGradient(0, y, 0, y + barH);
    if (index === 0) { gradient.addColorStop(0, "#f09aab"); gradient.addColorStop(1, "#ee7f98"); }
    else { gradient.addColorStop(0, "#777ace"); gradient.addColorStop(1, "#5559b8"); }
    roundRect(ctx, x, y, barW, barH, 8, gradient);
    ctx.fillStyle = "#6f7187";
    ctx.textAlign = "center";
    if (components) ctx.fillText(value.toFixed(1), x + barW / 2, y - 7);
    ctx.fillStyle = "#9899ab";
    ctx.fillText(names[index], x + barW / 2, height - 14);
  });
  ctx.textAlign = "start";
}

function roundRect(ctx, x, y, width, height, radius, fill) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r); ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r); ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

$("smilesInput").addEventListener("input", updateCharCount);
$("smilesInput").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") predict();
});
$("predictButton").addEventListener("click", predict);
$("copyButton").addEventListener("click", async () => {
  if (!state.result) return;
  await navigator.clipboard.writeText(state.result.canonical_smiles);
  $("copyButton").textContent = "已复制";
  setTimeout(() => { $("copyButton").textContent = "复制"; }, 1200);
});
window.addEventListener("resize", () => drawBranchChart(state.result?.components || null));

initialize().catch((error) => showError(`页面初始化失败：${error.message}`));
