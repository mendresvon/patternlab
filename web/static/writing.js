/* Safe text rendering and one request at a time. No sentence or result persistence. */
(() => {
  "use strict";
  const STATUSES = ["ok", "no_target_error", "abstained", "invalid_input", "unsupported_prediction", "generation_failed"];

  function editParts(text, edit, kind) {
    // Python offsets count Unicode code points; JS string.slice() counts UTF-16 units.
    if (!edit || !Array.isArray(edit.span) || edit.span.length !== 2 || typeof edit.replacement !== "string") throw new Error("Invalid edit");
    const [start, end] = edit.span;
    const points = Array.from(text);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end > points.length ||
        (kind !== "span" && kind !== "insertion") || (kind === "insertion") !== (start === end)) throw new Error("Invalid span");
    const before = points.slice(0, start).join("");
    const removed = points.slice(start, end).join("");
    const after = points.slice(end).join("");
    const corrected = before + edit.replacement + after;
    if (corrected === text || !corrected.trim()) throw new Error("No usable correction");
    return { before, removed, after, replacement: edit.replacement, corrected };
  }

  function validateResult(data, text, mode, categories) {
    if (!data || data.mode !== mode || !STATUSES.includes(data.status) ||
        (data.original_text !== text && !(data.status === "invalid_input" && data.original_text === null))) throw new Error("Invalid response");
    if (data.status !== "ok") {
      if (["category", "corrected_text", "edit", "location_kind", "explanation_zh"].some(key => data[key] !== null)) throw new Error("Unsafe failure response");
      return null;
    }
    if (!Object.hasOwn(categories, data.category) || typeof data.explanation_zh !== "string" || !/[\u3400-\u9fff]/.test(data.explanation_zh) || data.reason_code !== null) throw new Error("Invalid success");
    const parts = editParts(text, data.edit, data.location_kind);
    if (data.corrected_text !== parts.corrected) throw new Error("Inconsistent correction");
    return parts;
  }

  // Pure helpers are also exercised by Node's built-in test runner, with no build step.
  if (typeof module !== "undefined" && module.exports) module.exports = { editParts, validateResult };
  if (typeof document === "undefined") return;
  const root = document.querySelector("#writing-lab");
  if (!root) return;
  const find = id => document.getElementById(id);
  const input = find("sentence-input");
  const submit = find("analyze-sentence");
  const form = find("writing-form");
  const result = find("diagnosis-result");
  const mode = root.dataset.mode;
  const limit = Number(root.dataset.maxLength);
  const categories = JSON.parse(find("category-metadata").textContent);
  const samples = Array.from(root.querySelectorAll("[data-sample-text]"));
  const bank = find("bank-sentence");
  const reviewerExplanation = root.dataset.reviewer === "true" ? find("diagnosis-explanation-en") : null;
  let busy = false;
  let current = null;

  const messages = {
    no_target_error: ["目前沒有偵測到系統支援的目標錯誤。", "這不代表句子完全正確。你仍可以選一個主題，繼續累積英文語感。"],
    abstained: ["這次的線索還不夠明確。", "目前無法提供有把握的診斷，因此不顯示推測的修正。試著用更短、更單純的一句英文。"],
    invalid_input: ["先調整一下輸入，再試一次。", `請輸入一句非空白的英文，最多 ${limit} 個字元。你的原句會完整保留，不會自動截斷。`],
    unsupported_prediction: ["這次的內容超出支援範圍。", "目前每次只處理一句英文中的一個支援類別目標錯誤。請簡化句子，或直接選擇文法練習。"],
    generation_failed: ["診斷服務暫時無法使用。", "目前無法提供經過驗證的修正。你的原句已保留，可以稍後再試，或先做文法練習。"],
  };

  function element(tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function resetResult() {
    current = null;
    result.hidden = true;
    find("original-sentence").replaceChildren();
    find("corrected-sentence").replaceChildren();
    find("diagnosis-explanation").textContent = "";
    if (reviewerExplanation) reviewerExplanation.textContent = "";
    find("targeted-practice").href = "/practice";
    find("copy-status").textContent = "";
    find("copy-correction").textContent = "複製修正";
    find("copy-correction").disabled = false;
  }

  function updateInput() {
    const length = Array.from(input.value).length;
    find("writing-count").textContent = `${length} / ${limit} 字`;
    find("writing-count").classList.toggle("over-limit", length > limit);
    find("input-error").hidden = true;
    input.setAttribute("aria-invalid", "false");
    resetResult();
  }

  function setBusy(value) {
    busy = value;
    input.disabled = submit.disabled = value;
    samples.forEach(button => { button.disabled = value; });
    if (bank) bank.disabled = value;
    find("retry-analysis").disabled = value;
    find("try-another").disabled = value;
    form.setAttribute("aria-busy", String(value));
    submit.textContent = value ? "正在讀取結果…" : "分析句子 →";
    find("analysis-loading").hidden = !value;
  }

  function showResult(data, parts, connectionError = false) {
    current = data;
    const success = data.status === "ok";
    find("diagnosis-success").hidden = !success;
    find("diagnosis-alternatives").hidden = success;
    find("retry-analysis").hidden = data.status !== "generation_failed";
    result.classList.toggle("is-success", success);
    find("result-kicker").textContent = success ? "A SMALL DISCOVERY" : "ONE STEP AT A TIME";
    let title, message;
    if (success) {
      title = "找到一個文法小線索。";
      message = "先看看修改前後的差別，再用練習把觀念留下來。";
      const original = find("original-sentence");
      const corrected = find("corrected-sentence");
      original.replaceChildren(document.createTextNode(parts.before));
      if (data.location_kind === "insertion") {
        const caret = element("span", "▏", "insertion-caret");
        caret.setAttribute("role", "img");
        caret.setAttribute("aria-label", "此處需要補上文字");
        caret.title = "此處需要補上文字";
        original.append(caret);
      } else original.append(element("mark", parts.removed, "original-error"));
      original.append(document.createTextNode(parts.after));
      corrected.replaceChildren(document.createTextNode(parts.before));
      if (parts.replacement) corrected.append(element("mark", parts.replacement));
      corrected.append(document.createTextNode(parts.after));
      find("insertion-label").hidden = data.location_kind !== "insertion";
      find("diagnosis-category").textContent = categories[data.category].name;
      find("diagnosis-category-en").textContent = categories[data.category].english;
      find("diagnosis-explanation").textContent = data.explanation_zh;
      if (reviewerExplanation) reviewerExplanation.textContent = typeof data.explanation_en === "string" ? data.explanation_en : "";
      find("targeted-practice").href = `/practice/${data.category}`;
    } else {
      [title, message] = messages[data.status];
      if (data.reason_code === "sample_not_found") {
        title = "這句還沒有預先編寫的示範結果。";
        message = "示範模式不會分析新的句子。請選取頁面中的範例原句，再按「分析句子」。";
      } else if (data.reason_code === "input_too_long") {
        message = "這句超過模型可以完整處理的長度。請改寫成更短的一句英文；原文沒有被截斷。";
      } else if (data.reason_code === "service_busy") {
        title = "系統忙碌中，請稍後再試。";
        message = "目前正在處理另一個診斷請求。你的原句已保留，請稍後按「再試分析」。";
      } else if (data.reason_code === "invalid_prediction") {
        title = "這次的診斷結果沒有通過檢查。";
        message = "系統無法提供經過驗證的診斷，因此不顯示推測的修正。你的原句已保留，可以稍後再試，或先做文法練習。";
      } else if (data.reason_code === "invalid_generation") {
        message = "這次的修正結果沒有通過檢查，因此不會顯示。你的原句已保留，可以再試一次，或先做文法練習。";
      } else if (data.reason_code === "model_loading") {
        message = "即時模型仍在載入中，約需數分鐘。你的原句已保留，請稍後按「再試分析」；文法練習可以照常使用。";
        watchModels();
      } else if (mode === "live" && data.status === "generation_failed" && !connectionError && root.dataset.modelReady === "true") {
        message = "目前無法完成這次的修正。你的原句已保留；不會顯示未通過檢查的內容，文法練習可以照常使用。";
      } else if (mode === "live" && data.status === "generation_failed" && !connectionError) {
        message = "目前尚未連接即時模型，因此無法分析或提供修正。你的原句已保留；文法練習可以照常使用。";
      } else if (connectionError) {
        title = "連線暫時出了點問題。";
        message = "尚未取得可用的診斷結果。你的原句已保留，請確認連線後再試一次。";
      }
    }
    find("diagnosis-title").textContent = title;
    find("diagnosis-message").textContent = message;
    result.hidden = false;
    find("diagnosis-title").focus({ preventScroll: true });
    result.scrollIntoView({ block: "nearest", behavior: "auto" });
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (busy) return;
    const text = input.value;
    const length = Array.from(text).length;
    if (!text.trim() || length > limit) {
      find("input-error").textContent = !text.trim() ? "先寫一句英文，再開始分析。" : `這句超過 ${limit} 個字元，請縮短後再試。原文沒有被截斷。`;
      find("input-error").hidden = false;
      input.setAttribute("aria-invalid", "true");
      input.focus();
      return;
    }
    resetResult();
    find("input-error").hidden = true;
    setBusy(true);
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 65000);
    try {
      const response = await fetch("/api/diagnose", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }), signal: controller.signal, cache: "no-store",
      });
      const data = await response.json();
      if (!response.ok && !((response.status === 503 && data.status === "generation_failed") ||
          (response.status === 422 && data.status === "invalid_input"))) throw new Error("Service failed");
      const parts = validateResult(data, text, mode, categories);
      showResult(data, parts);
    } catch (error) {
      showResult({ status: "generation_failed", mode, reason_code: "connection_failed" }, null, true);
    } finally {
      window.clearTimeout(timer);
      setBusy(false);
    }
  });

  input.addEventListener("input", updateInput);
  if (bank) bank.addEventListener("change", () => {
    if (busy || !bank.value) return;
    input.value = bank.value;
    bank.value = "";
    updateInput();
    input.focus();
  });
  samples.forEach(button => button.addEventListener("click", () => {
    if (busy) return;
    input.value = button.dataset.sampleText;
    updateInput();
    input.focus();
  }));
  find("try-another").addEventListener("click", () => {
    if (busy) return;
    input.value = "";
    updateInput();
    input.focus();
  });
  find("retry-analysis").addEventListener("click", () => { if (!busy) form.requestSubmit(); });
  find("copy-correction").addEventListener("click", async () => {
    if (!current || current.status !== "ok") return;
    const copying = current;
    const button = find("copy-correction");
    if (button.disabled) return;
    button.disabled = true;
    try {
      await navigator.clipboard.writeText(copying.corrected_text);
      if (current === copying) {
        button.textContent = "已複製 ✓";
        find("copy-status").textContent = "修正後的句子已複製。";
        window.setTimeout(() => { if (current === copying) button.textContent = "複製修正"; }, 2500);
      }
    } catch (error) {
      if (current === copying) find("copy-status").textContent = "無法自動複製，請選取上方的修正句手動複製。";
    } finally {
      if (current === copying) button.disabled = false;
    }
  });
  // While the models warm up, check /health every 15 s and update the status in place.
  // Stop after about 10 minutes, so a forgotten tab cannot keep the GPU awake.
  // A tab opened before the service scaled to zero learns of the new warm-up from
  // the model_loading reason, and starts checking then.
  function showModelState(ready) {
    root.dataset.modelLoading = "false";
    root.dataset.modelReady = String(ready);
    find("service-indicator").classList.toggle("is-ready", ready);
    find("service-indicator-text").textContent = ready ? "即時診斷已就緒" : "即時診斷尚未連接";
    const availability = find("live-availability");
    if (availability) availability.textContent = ready
      ? "即時模型已連接，支援七大類別。只顯示通過檢查的修正；沒有偵測到目標錯誤，也不代表句子完全正確。"
      : "目前尚未連接診斷服務。送出時會如實顯示無法分析，不會產生範例修正。";
  }

  let checksLeft = 0;
  async function checkModels() {
    checksLeft -= 1;
    try {
      const response = await fetch("/health", { cache: "no-store" });
      const health = await response.json();
      if (response.ok && typeof health.model_loading === "boolean" && (health.model_ready === true || !health.model_loading)) {
        checksLeft = 0;
        return showModelState(health.model_ready === true);
      }
    } catch (error) { /* Try again on the next check. */ }
    if (checksLeft > 0) window.setTimeout(checkModels, 15000);
  }
  function watchModels() {
    root.dataset.modelLoading = "true";
    root.dataset.modelReady = "false";
    find("service-indicator").classList.toggle("is-ready", false);
    find("service-indicator-text").textContent = "模型載入中，約需數分鐘";
    const availability = find("live-availability");
    if (availability) availability.textContent = "即時模型正在載入，約需數分鐘。載入完成後，本頁會自動更新。";
    const idle = checksLeft <= 0;
    checksLeft = 40;
    if (idle) window.setTimeout(checkModels, 15000);
  }
  if (mode === "live" && root.dataset.modelLoading === "true") watchModels();

  submit.hidden = false;
  updateInput();
})();
