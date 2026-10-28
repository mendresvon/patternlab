/* Small progressive enhancements. The catalogue and all page links work without JS. */
(() => {
  "use strict";
  const STORAGE_KEY = "patternlab.progress.v1";
  const navShell = document.querySelector(".nav-shell");
  const menuToggle = document.querySelector(".menu-toggle");

  function closeMenu() {
    navShell.classList.remove("menu-open");
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.setAttribute("aria-label", "開啟導覽選單");
  }

  navShell.classList.add("menu-ready");
  const mobile = window.matchMedia("(max-width: 560px)");
  function syncMenu() {
    menuToggle.hidden = !mobile.matches;
    closeMenu();
  }
  syncMenu();
  mobile.addEventListener("change", syncMenu);
  menuToggle.addEventListener("click", () => {
    const open = navShell.classList.toggle("menu-open");
    menuToggle.setAttribute("aria-expanded", String(open));
    menuToggle.setAttribute("aria-label", open ? "關閉導覽選單" : "開啟導覽選單");
  });
  navShell.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && navShell.classList.contains("menu-open")) {
      closeMenu();
      menuToggle.focus();
    }
  });

  // The Python mapping also supplies the browser's category IDs and labels.
  const categories = JSON.parse(document.querySelector("#category-metadata").textContent);
  const categoryNames = Object.fromEntries(Object.entries(categories).map(([id, category]) => [id, category.name]));

  function validSession(session) {
    return session && typeof session.id === "string" && session.id.length > 0 && session.id.length <= 200 &&
      Object.hasOwn(categoryNames, session.category) && Number.isInteger(session.correct) &&
      session.total === 5 && session.correct >= 0 && session.correct <= session.total &&
      typeof session.completed_at === "string" && Number.isFinite(Date.parse(session.completed_at));
  }

  function readProgress() {
    // Only complete, valid five-question rounds contribute to progress.
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { sessions: [], notice: "", writable: true };
      const saved = JSON.parse(raw);
      if (!saved || saved.version !== 1 || !Array.isArray(saved.sessions)) throw new Error("Invalid progress");
      const seen = new Set();
      const sessions = saved.sessions.filter((session) => {
        if (!validSession(session) || seen.has(session.id)) return false;
        seen.add(session.id);
        return true;
      }).sort((a, b) => Date.parse(a.completed_at) - Date.parse(b.completed_at));
      const clean = sessions.length === saved.sessions.length;
      return { sessions, writable: clean, notice: clean ? "" : "部分進度格式不完整或重複，已略過；原始紀錄未變更。" };
    } catch (error) {
      return { sessions: [], writable: false, notice: "目前無法讀取已儲存的進度。你仍可練習，原始紀錄未變更。" };
    }
  }

  function saveCompletedSession(session) {
    if (!validSession(session)) return { saved: false, message: "這次紀錄不完整，未儲存。" };
    const { sessions, writable } = readProgress();
    if (sessions.some((item) => item.id === session.id)) return { saved: true, message: "進度已儲存在這個瀏覽器。" };
    if (!writable) return { saved: false, message: "無法安全讀取舊進度，這次未儲存。可先到首頁重設進度，原始資料未變更。" };
    try {
      // Store summaries only, never learner answers or question text.
      const summary = { id: session.id, category: session.category, correct: session.correct,
        total: 5, completed_at: session.completed_at };
      const completed = [...sessions, summary];
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1,
        xp: completed.reduce((sum, item) => sum + item.correct * 10, 0), sessions: completed }));
      renderProgress();
      return { saved: true, message: "進度已儲存在這個瀏覽器。" };
    } catch (error) {
      return { saved: false, message: "瀏覽器目前無法儲存進度。本次結果仍可查看，可再試一次。" };
    }
  }
  window.PatternLabProgress = { saveCompletedSession };

  function renderProgress() {
    const { sessions, notice } = readProgress();
    const correct = sessions.reduce((sum, session) => sum + session.correct, 0);
    const total = sessions.reduce((sum, session) => sum + session.total, 0);
    const stats = { sessions: sessions.length, xp: correct * 10, accuracy: total ? `${Math.round(correct / total * 100)}%` : "-" };
    document.querySelectorAll("[data-stat]").forEach((element) => { element.textContent = stats[element.dataset.stat]; });
    const storageNote = document.querySelector("[data-storage-note]");
    if (storageNote) { storageNote.textContent = notice; storageNote.hidden = !notice; }
    const accuracyCaption = document.querySelector("[data-accuracy-caption]");
    if (accuracyCaption) accuracyCaption.textContent = total ? `共 ${total} 題的首次作答紀錄` : "完成第一次練習後顯示";
    const latest = sessions.at(-1);
    const continueTitle = document.querySelector("[data-continue-title]");
    if (continueTitle) {
      document.querySelector("[data-continue-eyebrow]").textContent = latest ? "PICK UP WHERE YOU LEFT OFF" : "A GOOD PLACE TO START";
      continueTitle.textContent = latest ? `回顧上次的主題：${categoryNames[latest.category]}` : "從 a 和 an 開始，找到語感。";
      document.querySelector("[data-continue-description]").textContent = latest ? `上次答對 ${latest.correct} / ${latest.total} 題。再看看例句，讓觀念更清楚。` : "小小的冠詞，讓句子更完整。先認識一個簡單又常用的觀念。";
      document.querySelector("[data-continue-link]").href = `/practice/${latest ? latest.category : "article_omission"}`;
    }
    document.querySelectorAll("[data-category-progress]").forEach((element) => {
      const recent = sessions.filter((session) => session.category === element.dataset.categoryProgress).at(-1);
      element.textContent = recent ? `上次答對 ${recent.correct} / ${recent.total} 題` : "開始練習";
    });
  }
  renderProgress();
  window.addEventListener("storage", (event) => { if (event.key === STORAGE_KEY || event.key === null) renderProgress(); });
  window.addEventListener("pageshow", renderProgress);

  const resetButton = document.querySelector(".reset-progress");
  const resetDialog = document.querySelector("#reset-dialog");
  if (resetButton && resetDialog) {
    resetButton.hidden = false;
    resetButton.addEventListener("click", () => { resetDialog.returnValue = "cancel"; resetDialog.showModal(); });
    resetDialog.addEventListener("close", () => {
      if (resetDialog.returnValue !== "reset") return;
      const toast = document.querySelector(".toast");
      try {
        localStorage.removeItem(STORAGE_KEY);
        renderProgress();
        toast.textContent = "學習進度已重設，從新的起點出發。";
      } catch (error) {
        toast.textContent = "瀏覽器不允許變更儲存資料，進度未重設。";
      }
      toast.hidden = false;
      window.setTimeout(() => { toast.hidden = true; }, 4500);
    });
  }
})();
