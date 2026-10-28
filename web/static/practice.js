/* One small state machine: lesson → loading → answer → feedback → results. */
(() => {
  "use strict";
  const root = document.querySelector("#practice-session");
  if (!root) return;
  const find = (id) => document.getElementById(id);
  const lesson = find("lesson");
  const start = find("start-practice");
  const again = find("practice-again");
  const fields = find("answer-fields");
  const check = find("check-answer");
  const next = find("next-question");
  let round = null;
  let busy = false;
  let checked = false;
  let completion = null;
  let roundSequence = 0;

  function roundId() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
    // HTTP on another device lacks randomUUID. These IDs only deduplicate local
    // progress, so a timestamp, random suffix and sequence are sufficient.
    return `round-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${++roundSequence}`;
  }

  function textElement(tag, text, className) {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
  }

  async function request(url, options = {}) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, cache: "no-store" });
      if (!response.ok) throw new Error("Practice service unavailable");
      return await response.json();
    } finally {
      window.clearTimeout(timer);
    }
  }

  function showQuestion() {
    const question = round.questions[round.responses.length];
    checked = false;
    fields.disabled = false;
    check.disabled = false;
    check.hidden = false;
    check.textContent = "檢查答案";
    next.hidden = true;
    find("answer-feedback").hidden = true;
    find("quiz-error").hidden = true;
    find("question-index").textContent = `第 ${round.responses.length + 1} / 5 題`;
    find("question-prompt").textContent = question.prompt_zh;
    find("question-sentence").textContent = question.sentence;
    Array.from(document.querySelectorAll(".quiz-steps > span")).forEach((step, index) => {
      step.classList.toggle("done", index < round.responses.length);
      step.classList.toggle("current", index === round.responses.length);
    });
    const controls = find("answer-controls");
    controls.replaceChildren();
    if (question.type === "multiple_choice") {
      question.options.forEach((option, index) => {
        const label = textElement("label", "", "choice-option");
        const input = document.createElement("input");
        input.type = "radio";
        input.name = "answer";
        input.value = option.id;
        input.required = true;
        label.append(input, textElement("span", String.fromCharCode(65 + index), "choice-letter"));
        const text = textElement("span", option.text);
        text.lang = /[\u3400-\u9fff]/.test(option.text) ? "zh-Hant" : "en";
        label.append(text);
        controls.append(label);
      });
    } else {
      const label = textElement("label", "你的答案", "fill-label");
      label.htmlFor = "typed-answer";
      const input = document.createElement("input");
      input.type = "text";
      input.id = "typed-answer";
      input.name = "answer";
      input.maxLength = 100;
      input.required = true;
      input.autocomplete = "off";
      input.spellcheck = false;
      input.autocapitalize = "none";
      input.lang = "en";
      input.setAttribute("aria-describedby", "fill-hint");
      const hint = textElement("p", "只填空格缺少的字，不用重寫整句。", "fill-hint");
      hint.id = "fill-hint";
      controls.append(label, input, hint);
    }
    find("quiz-title").focus();
  }

  async function startRound() {
    if (busy) return;
    busy = true;
    start.disabled = again.disabled = true;
    start.textContent = again.textContent = "正在準備題目…";
    start.setAttribute("aria-busy", "true");
    find("round-error").hidden = true;
    try {
      const data = await request(`/api/practice/${root.dataset.category}/round`);
      if (data.category !== root.dataset.category || !Array.isArray(data.questions) || data.questions.length !== 5) throw new Error("Invalid round");
      round = { id: roundId(), questions: data.questions, responses: [] };
      completion = null;
      lesson.hidden = true;
      root.hidden = false;
      find("quiz-workspace").hidden = false;
      find("quiz-results").hidden = true;
      showQuestion();
    } catch (error) {
      find("round-error").textContent = "暫時無法取得題目，請確認連線後再試一次。課程仍可閱讀。";
      find("round-error").hidden = false;
    } finally {
      busy = false;
      start.disabled = again.disabled = false;
      start.textContent = "開始練習 →";
      again.textContent = "再練習一次 →";
      start.setAttribute("aria-busy", "false");
    }
  }

  find("answer-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy || checked || !round) return;
    const question = round.questions[round.responses.length];
    const answerInput = question.type === "multiple_choice" ? root.querySelector('input[name="answer"]:checked') : find("typed-answer");
    const answer = answerInput ? answerInput.value : "";
    if (!answer.trim()) {
      find("quiz-error").textContent = "先選一個答案，或填入空格需要的字。";
      find("quiz-error").hidden = false;
      return;
    }
    busy = true;
    fields.disabled = true;
    check.disabled = true;
    check.textContent = "正在檢查…";
    check.setAttribute("aria-busy", "true");
    find("quiz-error").hidden = true;
    try {
      const result = await request("/api/practice/check", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: question.id, answer }),
      });
      if (typeof result.correct !== "boolean" || typeof result.correct_answer !== "string" || typeof result.explanation_zh !== "string") throw new Error("Invalid feedback");
      checked = true;
      const userAnswer = question.type === "multiple_choice" ? question.options.find((option) => option.id === answer).text : answer.trim();
      round.responses.push({ question, userAnswer, ...result });
      find("feedback-title").textContent = result.correct ? "✓ 答對了！這個觀念抓住了。" : "再看一次，就更清楚了。";
      find("feedback-answer").textContent = `正確答案：${result.correct_answer}`;
      find("feedback-explanation").textContent = result.explanation_zh;
      find("answer-feedback").classList.toggle("incorrect", !result.correct);
      find("answer-feedback").hidden = false;
      check.hidden = true;
      next.textContent = round.responses.length === 5 ? "查看練習成果 →" : "下一題 →";
      next.hidden = false;
      next.focus();
    } catch (error) {
      fields.disabled = false;
      check.disabled = false;
      check.textContent = "再試檢查答案";
      find("quiz-error").textContent = "暫時無法檢查答案，尚未計分。答案已保留，請再試一次。";
      find("quiz-error").hidden = false;
    } finally {
      busy = false;
      check.setAttribute("aria-busy", "false");
    }
  });

  function saveResult() {
    const result = window.PatternLabProgress.saveCompletedSession(completion);
    find("save-status").textContent = result.message;
    find("retry-save").hidden = result.saved;
  }

  function showResults() {
    const correct = round.responses.filter((answer) => answer.correct).length;
    find("quiz-workspace").hidden = true;
    find("quiz-results").hidden = false;
    find("result-score").textContent = `${correct}/5`;
    find("result-accuracy").textContent = `${correct * 20}%`;
    find("result-xp").textContent = `+${correct * 10} XP`;
    const missed = round.responses.filter((answer) => !answer.correct);
    const review = find("missed-items");
    review.replaceChildren();
    find("review-title").textContent = missed.length ? "把這幾題，再看懂一次。" : "全對！把這份語感帶進下一句英文。";
    for (const answer of missed) {
      const item = textElement("article", "", "review-item");
      item.append(textElement("h4", answer.question.prompt_zh));
      const sentence = textElement("p", answer.question.sentence, "review-sentence");
      sentence.lang = "en";
      item.append(sentence, textElement("p", `你的答案：${answer.userAnswer}`),
        textElement("p", `正確答案：${answer.correct_answer}`, "review-correct"),
        textElement("p", answer.explanation_zh));
      review.append(item);
    }
    if (!completion) completion = { id: round.id, category: root.dataset.category, correct,
      total: 5, completed_at: new Date().toISOString() };
    saveResult();
    find("results-title").focus();
  }

  next.addEventListener("click", () => {
    if (busy || !checked) return;
    if (round.responses.length === 5) showResults();
    else showQuestion();
  });
  find("back-to-lesson").addEventListener("click", () => {
    if (busy) return;
    round = null;
    root.hidden = true;
    lesson.hidden = false;
    start.focus();
  });
  start.hidden = false;
  start.addEventListener("click", startRound);
  again.addEventListener("click", startRound);
  find("retry-save").addEventListener("click", saveResult);
})();
