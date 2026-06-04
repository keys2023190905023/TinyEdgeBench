const operatorFamilies = [
  "conv2d",
  "depthwise_conv2d",
  "pointwise_conv2d",
  "matmul",
  "batch_matmul",
  "linear",
  "relu",
  "gelu",
  "silu",
  "hard_swish",
  "mish",
  "softmax",
  "layernorm",
  "rmsnorm",
  "groupnorm",
  "instance_norm",
  "maxpool2d",
  "avgpool2d",
  "global_avgpool2d",
  "add",
  "mul",
  "concat",
  "gather",
  "one_hot",
  "space_to_depth",
  "depth_to_space",
  "scaled_dot_product_attention",
  "causal_self_attention",
  "rotary_embedding",
  "cosine_similarity",
  "topk",
  "sort",
  "cumsum",
  "swiglu",
  "geglu",
];

const cloud = document.querySelector("#operator-cloud");
if (cloud) {
  operatorFamilies.forEach((name) => {
    const chip = document.createElement("span");
    chip.className = "operator-chip";
    chip.textContent = name;
    cloud.appendChild(chip);
  });
}

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const body = document.body;
const cursorRing = document.querySelector(".cursor-ring");
const runtimeSearchParams = new URLSearchParams(window.location.search);
const requestedMotionMode = runtimeSearchParams.get("motion");
const motionProfile = {
  performanceLite: !prefersReducedMotion.matches && requestedMotionMode !== "full",
};

body.classList.toggle("is-performance-lite", motionProfile.performanceLite);

function getCanvasRatio(standardLimit = 1.5, liteLimit = 1) {
  return Math.min(window.devicePixelRatio || 1, motionProfile.performanceLite ? liteLimit : standardLimit);
}

function getMotionFrameInterval(baseInterval, liteInterval = baseInterval * 1.4, scrollingInterval = liteInterval * 1.6) {
  if (body.classList.contains("is-scrolling")) return scrollingInterval;
  return motionProfile.performanceLite ? liteInterval : baseInterval;
}

function isPerformanceLiteScrollActive() {
  return motionProfile.performanceLite && body.classList.contains("is-scrolling");
}

const liveRuns = [
  {
    label: "conv2d / torch_cpu / int8_sim",
    latency: 2.84,
    throughput: 352,
    error: 0.0031,
    backend: "torch_cpu",
    fastest: "torch_cuda",
    bottleneck: "conv2d",
    progress: 68,
    bars: { cpu: 4.92, torch: 2.84, onnx: 2.31, cuda: 0.71 },
  },
  {
    label: "matmul / onnxruntime_cpu / fp32",
    latency: 1.76,
    throughput: 568,
    error: 0.0,
    backend: "onnxruntime_cpu",
    fastest: "onnxruntime_cpu",
    bottleneck: "matmul",
    progress: 82,
    bars: { cpu: 3.68, torch: 2.12, onnx: 1.76, cuda: 0.64 },
  },
  {
    label: "depthwise_conv2d / cpu / shift_only",
    latency: 3.18,
    throughput: 314,
    error: 0.0184,
    backend: "cpu",
    fastest: "torch_cpu",
    bottleneck: "depthwise_conv2d",
    progress: 54,
    bars: { cpu: 3.18, torch: 2.43, onnx: 2.57, cuda: 0.88 },
  },
  {
    label: "llama_mlp_block / torch_cuda / fp32",
    latency: 0.93,
    throughput: 1075,
    error: 0.0,
    backend: "torch_cuda",
    fastest: "torch_cuda",
    bottleneck: "swiglu",
    progress: 91,
    bars: { cpu: 8.42, torch: 4.31, onnx: 3.92, cuda: 0.93 },
  },
];

const liveEls = {
  label: document.querySelector("#live-run-label"),
  progressLabel: document.querySelector("#run-progress-label"),
  progressBar: document.querySelector("#run-progress-bar"),
  latency: document.querySelector("#live-latency"),
  throughput: document.querySelector("#live-throughput"),
  error: document.querySelector("#live-error"),
  backend: document.querySelector("#live-backend"),
  fastest: document.querySelector("#fastest-backend"),
  bottleneck: document.querySelector("#bottleneck-op"),
  cpuBar: document.querySelector("#bar-cpu"),
  torchBar: document.querySelector("#bar-torch"),
  onnxBar: document.querySelector("#bar-onnx"),
  cudaBar: document.querySelector("#bar-cuda"),
  cpuValue: document.querySelector("#value-cpu"),
  torchValue: document.querySelector("#value-torch"),
  onnxValue: document.querySelector("#value-onnx"),
  cudaValue: document.querySelector("#value-cuda"),
};

function setText(node, value) {
  if (node) node.textContent = value;
}

function setBar(node, value, max) {
  if (!node) return;
  const widthValue = Math.max(12, Math.round((value / max) * 100));
  node.style.width = `${widthValue}%`;
}

function updateLiveRun(run) {
  const maxLatency = Math.max(run.bars.cpu, run.bars.torch, run.bars.onnx, run.bars.cuda);
  setText(liveEls.label, run.label);
  setText(liveEls.progressLabel, `${run.progress}%`);
  setText(liveEls.latency, `${run.latency.toFixed(2)} ms`);
  setText(liveEls.throughput, `${run.throughput} ops/s`);
  setText(liveEls.error, run.error.toFixed(4));
  setText(liveEls.backend, run.backend);
  setText(liveEls.fastest, run.fastest);
  setText(liveEls.bottleneck, run.bottleneck);
  setText(liveEls.cpuValue, `${run.bars.cpu.toFixed(2)} ms`);
  setText(liveEls.torchValue, `${run.bars.torch.toFixed(2)} ms`);
  setText(liveEls.onnxValue, `${run.bars.onnx.toFixed(2)} ms`);
  setText(liveEls.cudaValue, `${run.bars.cuda.toFixed(2)} ms`);
  if (liveEls.progressBar) liveEls.progressBar.style.width = `${run.progress}%`;
  setBar(liveEls.cpuBar, run.bars.cpu, maxLatency);
  setBar(liveEls.torchBar, run.bars.torch, maxLatency);
  setBar(liveEls.onnxBar, run.bars.onnx, maxLatency);
  setBar(liveEls.cudaBar, run.bars.cuda, maxLatency);
}

function initializeLivePreview() {
  if (!liveEls.label) return;
  let liveIndex = 0;
  updateLiveRun(liveRuns[liveIndex]);
  if (motionProfile.performanceLite) return;
  window.setInterval(() => {
    liveIndex = (liveIndex + 1) % liveRuns.length;
    updateLiveRun(liveRuns[liveIndex]);
  }, 2400);
}

function tagRevealItems(selector, options = {}) {
  const { baseDelay = 0, step = 90, style = "" } = options;
  document.querySelectorAll(selector).forEach((element, index) => {
    element.classList.add("reveal-item");
    if (style) element.classList.add(`reveal-${style}`);
    element.style.setProperty("--reveal-delay", `${baseDelay + index * step}ms`);
  });
}

function initializeRevealAnimations() {
  tagRevealItems(".hero-copy-block > *", { baseDelay: 40, step: 90, style: "rotate" });
  tagRevealItems(".hero-console-shell > *", { baseDelay: 220, step: 90 });
  tagRevealItems(".result-card", { baseDelay: 0, step: 100 });
  tagRevealItems(".manifesto-copy > *", { baseDelay: 0, step: 90, style: "soft" });
  tagRevealItems(".manifesto-grid > div", { baseDelay: 60, step: 120 });
  tagRevealItems(".proof-grid > div", { baseDelay: 0, step: 110 });
  tagRevealItems(".feature-card", { baseDelay: 0, step: 90 });
  tagRevealItems(".run-monitor, .latency-panel", { baseDelay: 0, step: 120 });
  tagRevealItems(".hardware-panel, .backend-matrix", { baseDelay: 0, step: 120 });
  tagRevealItems(".story-heading > *", { baseDelay: 0, step: 80, style: "soft" });
  tagRevealItems(".catalog-card", { baseDelay: 0, step: 90 });
  tagRevealItems(".protocol-flow > div", { baseDelay: 0, step: 80 });
  tagRevealItems(".workflow-step", { baseDelay: 0, step: 80 });
  tagRevealItems(".install-copy > *", { baseDelay: 0, step: 90, style: "soft" });
  tagRevealItems(".terminal-window", { baseDelay: 180, step: 0 });

  const revealItems = document.querySelectorAll(".reveal-item");
  document.querySelectorAll(".hero .reveal-item").forEach((item) => item.classList.add("is-visible"));
  if (prefersReducedMotion.matches || motionProfile.performanceLite) {
    revealItems.forEach((item) => item.classList.add("is-visible"));
    return;
  }

  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    {
      threshold: 0.16,
      rootMargin: "0px 0px -10% 0px",
    },
  );

  revealItems.forEach((item) => revealObserver.observe(item));
}

function initializeActiveNav() {
  const navLinks = [...document.querySelectorAll('.nav-links a[href^="#"]')];
  if (!navLinks.length) return;

  const sectionMap = new Map();
  const sections = navLinks
    .map((link) => {
      const id = link.getAttribute("href").slice(1);
      const section = document.getElementById(id);
      if (!section) return null;
      sectionMap.set(id, 0);
      return section;
    })
    .filter(Boolean);

  const navObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        sectionMap.set(entry.target.id, entry.intersectionRatio);
      });

      let activeId = "";
      let maxRatio = 0.12;
      sectionMap.forEach((ratio, id) => {
        if (ratio > maxRatio) {
          maxRatio = ratio;
          activeId = id;
        }
      });

      navLinks.forEach((link) => {
        const linkId = link.getAttribute("href").slice(1);
        link.classList.toggle("is-active", linkId === activeId);
      });
    },
    {
      threshold: [0.15, 0.35, 0.6],
      rootMargin: "-18% 0px -58% 0px",
    },
  );

  sections.forEach((section) => navObserver.observe(section));
}

function initializeScrollState() {
  if (motionProfile.performanceLite) {
    let scrollIdleTimer = 0;

    const updateScrollLite = () => {
      body.classList.toggle("is-scrolled", window.scrollY > 28);
    };

    const handleLiteScroll = () => {
      body.classList.add("is-scrolling");
      window.clearTimeout(scrollIdleTimer);
      scrollIdleTimer = window.setTimeout(() => {
        body.classList.remove("is-scrolling");
      }, 120);
      updateScrollLite();
    };

    window.addEventListener("scroll", handleLiteScroll, { passive: true });
    updateScrollLite();
    return;
  }

  let scrollIdleTimer = 0;

  const updateScroll = () => {
    const ratio = Math.min(window.scrollY / Math.max(window.innerHeight, 1), 1);
    body.style.setProperty("--scroll-progress", ratio.toFixed(3));
    body.classList.toggle("is-scrolled", window.scrollY > 28);
  };

  let ticking = false;
  const handleScroll = () => {
    body.classList.add("is-scrolling");
    window.clearTimeout(scrollIdleTimer);
    scrollIdleTimer = window.setTimeout(() => {
      body.classList.remove("is-scrolling");
    }, 140);

    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(() => {
      updateScroll();
      ticking = false;
    });
  };

  window.addEventListener("scroll", handleScroll, { passive: true });
  updateScroll();
}

function initializeHeroParallax() {
  const hero = document.querySelector(".hero");
  if (!hero || prefersReducedMotion.matches) return;

  const state = {
    copyX: 0,
    copyY: 0,
    panelX: 0,
    panelY: 0,
    glowX: 0,
    glowY: 0,
    targetX: 0,
    targetY: 0,
  };

  const updateTarget = (event) => {
    const rect = hero.getBoundingClientRect();
    const insideX = event.clientX >= rect.left && event.clientX <= rect.right;
    const insideY = event.clientY >= rect.top && event.clientY <= rect.bottom;
    if (!insideX || !insideY) {
      state.targetX = 0;
      state.targetY = 0;
      return;
    }
    const offsetX = (event.clientX - rect.left) / rect.width - 0.5;
    const offsetY = (event.clientY - rect.top) / rect.height - 0.5;
    state.targetX = offsetX;
    state.targetY = offsetY;
  };

  window.addEventListener("pointermove", updateTarget);
  window.addEventListener("pointerleave", () => {
    state.targetX = 0;
    state.targetY = 0;
  });
  window.addEventListener("blur", () => {
    state.targetX = 0;
    state.targetY = 0;
  });

  const render = () => {
    state.copyX += ((state.targetX * -16) - state.copyX) * 0.08;
    state.copyY += ((state.targetY * -12) - state.copyY) * 0.08;
    state.panelX += ((state.targetX * 18) - state.panelX) * 0.08;
    state.panelY += ((state.targetY * 16) - state.panelY) * 0.08;
    state.glowX += ((state.targetX * 26) - state.glowX) * 0.08;
    state.glowY += ((state.targetY * 20) - state.glowY) * 0.08;

    body.style.setProperty("--hero-copy-x", `${state.copyX.toFixed(2)}px`);
    body.style.setProperty("--hero-copy-y", `${state.copyY.toFixed(2)}px`);
    body.style.setProperty("--hero-panel-x", `${state.panelX.toFixed(2)}px`);
    body.style.setProperty("--hero-panel-y", `${state.panelY.toFixed(2)}px`);
    body.style.setProperty("--hero-glow-x", `${state.glowX.toFixed(2)}px`);
    body.style.setProperty("--hero-glow-y", `${state.glowY.toFixed(2)}px`);

    window.requestAnimationFrame(render);
  };

  render();
}

function initializeInitialSectionJump() {
  const resolveTargetId = () => {
    const searchParams = new URLSearchParams(window.location.search);
    const hashTarget = window.location.hash ? window.location.hash.slice(1) : "";
    const paramTarget = searchParams.get("section") || "";
    return hashTarget || paramTarget;
  };

  const jumpToTarget = () => {
    const targetId = resolveTargetId();
    if (!targetId) return false;

    const target = document.getElementById(targetId);
    if (!target) return false;

    const headerHeight = document.querySelector(".site-header")?.offsetHeight ?? 0;
    const offset = Math.max(24, headerHeight - 12);
    const top = Math.max(0, target.getBoundingClientRect().top + window.scrollY - offset);
    window.scrollTo({ top, behavior: "auto" });
    return true;
  };

  if (!resolveTargetId()) return;

  [36, 160, 360, 720].forEach((delay) => {
    window.setTimeout(jumpToTarget, delay);
  });

  window.addEventListener(
    "load",
    () => {
      window.setTimeout(jumpToTarget, 64);
    },
    { once: true },
  );

  window.addEventListener("hashchange", () => {
    window.setTimeout(jumpToTarget, 24);
  });
}

function initializeTurnSections() {
  const sections = [...document.querySelectorAll("[data-turn-section]")];
  if (!sections.length) return;

  if (motionProfile.performanceLite) {
    const phaseConfig = {
      idle: {
        progress: 0.78,
        entry: 0.84,
        dissolve: 0,
        visibility: 0.74,
        cloud: 0.12,
        shift: 18,
        blur: 5,
        scale: 0.986,
        glow: 0.14,
        curl: 0.16,
        assembly: 0.88,
      },
      previous: {
        progress: 0.9,
        entry: 1,
        dissolve: 0.12,
        visibility: 0.84,
        cloud: 0.18,
        shift: -12,
        blur: 1,
        scale: 0.992,
        glow: 0.18,
        curl: 0.18,
        assembly: 1,
      },
      active: {
        progress: 1,
        entry: 1,
        dissolve: 0,
        visibility: 1,
        cloud: 0.28,
        shift: 0,
        blur: 0,
        scale: 1,
        glow: 0.24,
        curl: 0.18,
        assembly: 1,
      },
      next: {
        progress: 0.94,
        entry: 0.95,
        dissolve: 0,
        visibility: 0.91,
        cloud: 0.22,
        shift: 8,
        blur: 1,
        scale: 0.994,
        glow: 0.18,
        curl: 0.16,
        assembly: 0.95,
      },
    };

    const applyPhase = (section, phase, index) => {
      const config = phaseConfig[phase] || phaseConfig.idle;
      section.style.setProperty("--turn-rotate", "0deg");
      section.style.setProperty("--turn-yaw", "0deg");
      section.style.setProperty("--turn-shift", "0px");
      section.style.setProperty("--turn-scale", `${config.scale}`);
      section.style.setProperty("--turn-glow", `${config.glow}`);
      section.style.setProperty("--turn-curl", `${config.curl}`);
      section.style.setProperty("--chapter-progress", `${config.progress}`);
      section.style.setProperty("--chapter-entry", `${config.entry}`);
      section.style.setProperty("--chapter-dissolve", `${config.dissolve}`);
      section.style.setProperty("--chapter-visibility", `${config.visibility}`);
      section.style.setProperty("--chapter-cloud", `${config.cloud}`);
      section.style.setProperty("--chapter-shift", `${config.shift}px`);
      section.style.setProperty("--chapter-blur", `${config.blur}px`);
      section.style.setProperty("--assembly-progress", `${config.assembly}`);
      section.style.setProperty("--chapter-z", `${sections.length - index}`);
      section.classList.toggle("is-chapter-live", phase === "active" || phase === "next");
      section.classList.toggle("is-chapter-settled", phase === "active");
      section.classList.toggle("is-chapter-dissolving", phase === "previous");
    };

    const visibilityMap = new Map(sections.map((section) => [section, 0]));

    const syncSections = () => {
      let activeIndex = 0;
      let maxRatio = -1;

      sections.forEach((section, index) => {
        const ratio = visibilityMap.get(section) ?? 0;
        if (ratio > maxRatio) {
          maxRatio = ratio;
          activeIndex = index;
        }
      });

      sections.forEach((section, index) => {
        let phase = "idle";
        if (index === activeIndex) phase = "active";
        else if (index === activeIndex - 1) phase = "previous";
        else if (index === activeIndex + 1) phase = "next";
        applyPhase(section, phase, index);
      });
    };

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          visibilityMap.set(entry.target, entry.intersectionRatio);
        });
        syncSections();
      },
      {
        threshold: [0, 0.2, 0.4, 0.6, 0.8, 1],
        rootMargin: "-12% 0px -18% 0px",
      },
    );

    sections.forEach((section, index) => {
      applyPhase(section, index === 0 ? "active" : index === 1 ? "next" : "idle", index);
      observer.observe(section);
    });
    return;
  }

  const clamp01 = (value) => Math.max(0, Math.min(1, value));
  const easeOutCubic = (value) => 1 - (1 - value) ** 3;
  const easeInOutCubic = (value) =>
    value < 0.5 ? 4 * value ** 3 : 1 - ((-2 * value + 2) ** 3) / 2;

  const update = () => {
    const viewportHeight = Math.max(window.innerHeight, 1);

    sections.forEach((section, index) => {
      const rect = section.getBoundingClientRect();
      const nextRect = sections[index + 1]?.getBoundingClientRect();
      const centerOffset = (rect.top + rect.height * 0.5 - viewportHeight * 0.56) / (viewportHeight * 0.92);
      const clampedOffset = Math.max(-1, Math.min(1, centerOffset));
      const focus = clamp01(1 - Math.abs(centerOffset));
      const chapterDirection = (Number.parseInt(section.dataset.turnSection || "1", 10) || 1) % 2 === 0 ? -1 : 1;
      const revealStart = viewportHeight * 1.14;
      const revealEnd = viewportHeight * 0.26;
      const rawEntry = prefersReducedMotion.matches
        ? 1
        : clamp01(1 - (rect.top - revealEnd) / Math.max(revealStart - revealEnd, 1));
      const chapterEntry = prefersReducedMotion.matches ? 1 : easeInOutCubic(rawEntry);
      const handoffStart = viewportHeight * 1.08;
      const handoffEnd = viewportHeight * 0.28;
      const rawDissolve = prefersReducedMotion.matches || !nextRect
        ? 0
        : clamp01(1 - (nextRect.top - handoffEnd) / Math.max(handoffStart - handoffEnd, 1));
      const chapterDissolve = prefersReducedMotion.matches ? 0 : easeInOutCubic(rawDissolve);
      const overlapWindow = prefersReducedMotion.matches
        ? 0.18
        : clamp01(1 - Math.abs(rawEntry - rawDissolve) / 0.5);
      const chapterVisibility = prefersReducedMotion.matches
        ? 1
        : clamp01(0.06 + chapterEntry * 1.16 - chapterDissolve * 0.34);
      const chapterCloud = prefersReducedMotion.matches
        ? 0.28
        : clamp01(
            0.08 +
              chapterEntry * 0.26 +
              rawDissolve * 0.82 +
              overlapWindow * 0.34,
          );
      const chapterProgress = prefersReducedMotion.matches
        ? 1
        : clamp01(chapterEntry * 0.78 + (1 - chapterDissolve) * 0.22);
      const assemblyMode = section.dataset.assemblyMode || "";
      const assemblyProgress = prefersReducedMotion.matches
        ? 1
        : easeOutCubic(
            clamp01(assemblyMode === "reassemble" ? (chapterEntry - 0.08) / 0.78 : chapterEntry),
          ) * (1 - chapterDissolve * 0.08);
      const rotate = prefersReducedMotion.matches
        ? 0
        : clampedOffset * -5.8 - (1 - chapterEntry) * 2.2 + chapterDissolve * 3.4;
      const yaw = prefersReducedMotion.matches
        ? 0
        : clampedOffset * chapterDirection * (2.9 + (1 - chapterEntry) * 2.2) + chapterDirection * chapterDissolve * 2.8;
      const shift = prefersReducedMotion.matches ? 0 : clampedOffset * 12 - chapterDissolve * 12;
      const scale = prefersReducedMotion.matches
        ? 1
        : 0.978 + focus * 0.014 + chapterEntry * 0.01 - chapterDissolve * 0.014;
      const curl = prefersReducedMotion.matches
        ? 0.16
        : 0.16 + Math.abs(clampedOffset) * 0.48 + chapterDissolve * 0.2;
      const chapterShift = prefersReducedMotion.matches ? 0 : (1 - chapterEntry) * 56 - chapterDissolve * 10;
      const blur = prefersReducedMotion.matches ? 0 : (1 - chapterEntry) * 14 + chapterDissolve * 8;
      const chapterZ = Math.max(1, sections.length - index - Math.round(chapterDissolve * 2.2));
      const isChapterLive = chapterEntry > 0.08 && chapterDissolve < 0.98;
      const isChapterSettled = chapterEntry >= 0.98 && chapterDissolve < 0.18;

      section.style.setProperty("--turn-rotate", `${rotate.toFixed(2)}deg`);
      section.style.setProperty("--turn-yaw", `${yaw.toFixed(2)}deg`);
      section.style.setProperty("--turn-shift", `${shift.toFixed(2)}px`);
      section.style.setProperty("--turn-scale", `${scale.toFixed(3)}`);
      section.style.setProperty("--turn-glow", `${focus.toFixed(3)}`);
      section.style.setProperty("--turn-curl", `${curl.toFixed(3)}`);
      section.style.setProperty("--chapter-progress", `${chapterProgress.toFixed(3)}`);
      section.style.setProperty("--chapter-entry", `${chapterEntry.toFixed(3)}`);
      section.style.setProperty("--chapter-dissolve", `${chapterDissolve.toFixed(3)}`);
      section.style.setProperty("--chapter-visibility", `${chapterVisibility.toFixed(3)}`);
      section.style.setProperty("--chapter-cloud", `${chapterCloud.toFixed(3)}`);
      section.style.setProperty("--chapter-shift", `${chapterShift.toFixed(2)}px`);
      section.style.setProperty("--chapter-blur", `${blur.toFixed(2)}px`);
      section.style.setProperty("--assembly-progress", `${assemblyProgress.toFixed(3)}`);
      section.style.setProperty("--chapter-z", `${chapterZ}`);
      section.classList.toggle("is-chapter-live", isChapterLive);
      section.classList.toggle("is-chapter-settled", isChapterSettled);
      section.classList.toggle("is-chapter-dissolving", chapterDissolve > 0.06);
    });
  };

  let ticking = false;
  const requestUpdate = () => {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(() => {
      update();
      ticking = false;
    });
  };

  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate);
  update();
}

function initializeManifestoCanvas() {
  const section = document.querySelector(".proof-band");
  const canvas = document.querySelector("#manifesto-canvas");
  const canvasCtx = canvas?.getContext("2d", { alpha: true });
  if (!section || !canvas || !canvasCtx) return;

  let canvasWidth = 0;
  let canvasHeight = 0;
  let active = true;
  let lastFrame = 0;

  const blobCount = 12;
  const blobs = Array.from({ length: blobCount }, (_, index) => {
    const leftSide = index < blobCount / 2;
    return {
      side: leftSide ? "left" : "right",
      phase: Math.random() * Math.PI * 2,
      speed: 0.28 + Math.random() * 0.34,
      radius: 120 + Math.random() * 170,
      offsetY: 0.12 + Math.random() * 0.72,
      sway: 18 + Math.random() * 46,
      drift: 10 + Math.random() * 34,
    };
  });

  function resizeManifestoCanvas() {
    const ratio = getCanvasRatio(1.2, 1);
    canvasWidth = section.clientWidth;
    canvasHeight = section.clientHeight;
    canvas.width = Math.max(1, Math.floor(canvasWidth * ratio));
    canvas.height = Math.max(1, Math.floor(canvasHeight * ratio));
    canvas.style.width = `${canvasWidth}px`;
    canvas.style.height = `${canvasHeight}px`;
    canvasCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  if (!prefersReducedMotion.matches) {
    const canvasObserver = new IntersectionObserver(
      (entries) => {
        active = entries.some((entry) => entry.isIntersecting);
      },
      { threshold: 0.08 },
    );
    canvasObserver.observe(section);
  }

  function drawCloud(timeSeconds) {
    canvasCtx.clearRect(0, 0, canvasWidth, canvasHeight);

    const cell = canvasWidth < 700 ? 11 : 13;
    const halfWidth = canvasWidth * 0.5;
    const verticalCenter = canvasHeight * 0.34;

    for (let y = 0; y < canvasHeight; y += cell) {
      for (let x = 0; x < canvasWidth; x += cell) {
        let intensity = 0;
        for (const blob of blobs) {
          const direction = blob.side === "left" ? 1 : -1;
          const anchorX = blob.side === "left" ? canvasWidth * 0.08 : canvasWidth * 0.92;
          const offsetX = Math.sin(timeSeconds * blob.speed + blob.phase) * blob.sway;
          const offsetY = Math.cos(timeSeconds * (blob.speed * 0.8) + blob.phase) * blob.drift;
          const blobX = anchorX + direction * (36 + Math.sin(timeSeconds * 0.18 + blob.phase) * 34) + offsetX;
          const blobY = canvasHeight * blob.offsetY + offsetY;
          const dx = x - blobX;
          const dy = y - blobY;
          intensity += (blob.radius * blob.radius) / (dx * dx * 1.1 + dy * dy * 0.95 + blob.radius * 18);
        }

        const centerFade = Math.abs(x - halfWidth) / halfWidth;
        const topFade = Math.abs(y - verticalCenter) / Math.max(canvasHeight * 0.42, 1);
        intensity *= Math.max(centerFade - 0.08, 0) * 1.6;
        intensity *= Math.max(1 - topFade * 0.55, 0.24);

        if (intensity < 0.9) continue;

        const shade = Math.min(255, Math.round(114 + intensity * 58));
        const alpha = Math.min(0.9, 0.12 + intensity * 0.12);
        canvasCtx.fillStyle = `rgba(${shade}, ${shade}, ${shade}, ${alpha})`;
        canvasCtx.fillRect(x, y, cell - 1, cell - 1);
      }
    }

    for (let star = 0; star < 26; star += 1) {
      const starX = ((star * 97.3) % canvasWidth);
      const starY = ((star * 53.7 + timeSeconds * 22) % canvasHeight);
      const brightness = 0.18 + ((Math.sin(timeSeconds * 0.9 + star) + 1) * 0.08);
      canvasCtx.fillStyle = `rgba(255, 255, 255, ${brightness})`;
      canvasCtx.fillRect(starX, starY, 1.5, 1.5);
    }
  }

  function renderManifesto(now) {
    if (!lastFrame || now - lastFrame > getMotionFrameInterval(42, 62, 92)) {
      if ((active || prefersReducedMotion.matches) && !isPerformanceLiteScrollActive()) {
        drawCloud(now / 1000);
      }
      lastFrame = now;
    }
    window.requestAnimationFrame(renderManifesto);
  }

  window.addEventListener("resize", resizeManifestoCanvas);
  resizeManifestoCanvas();
  drawCloud(0);
  if (!prefersReducedMotion.matches) {
    window.requestAnimationFrame(renderManifesto);
  }
}

function initializeStoryCanvas() {
  const stage = document.querySelector(".story-stage");
  const stack = stage?.querySelector(".story-stack");
  const canvas = document.querySelector("#story-canvas");
  const canvasCtx = canvas?.getContext("2d", { alpha: true });
  if (!stage || !stack || !canvas || !canvasCtx) return;

  const narrativeSections = [...stack.querySelectorAll(".section")];
  if (!narrativeSections.length) return;

  let canvasWidth = 0;
  let canvasHeight = 0;
  let active = true;
  let lastFrame = 0;

  function resizeStoryCanvas() {
    const ratio = getCanvasRatio(1.15, 1);
    canvasWidth = stage.clientWidth;
    canvasHeight = stage.clientHeight;
    canvas.width = Math.max(1, Math.floor(canvasWidth * ratio));
    canvas.height = Math.max(1, Math.floor(canvasHeight * ratio));
    canvas.style.width = `${canvasWidth}px`;
    canvas.style.height = `${canvasHeight}px`;
    canvasCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function buildAnchors(timeSeconds) {
    return narrativeSections.map((section, index) => {
      const baseY = section.offsetTop + Math.min(section.clientHeight * 0.34, 220);
      const side = index % 2 === 0 ? "right" : "left";
      const baseX = side === "right" ? canvasWidth * 0.82 : canvasWidth * 0.18;
      const sway = Math.sin(timeSeconds * (0.45 + index * 0.06) + index) * 24;
      const driftY = Math.cos(timeSeconds * (0.32 + index * 0.04) + index * 1.3) * 16;
      return {
        x: baseX + sway,
        y: baseY + driftY,
        side,
        phase: index * 1.4,
        radius: 170 + index * 18,
      };
    });
  }

  if (!prefersReducedMotion.matches) {
    const stageObserver = new IntersectionObserver(
      (entries) => {
        active = entries.some((entry) => entry.isIntersecting);
      },
      { threshold: 0.08 },
    );
    stageObserver.observe(stage);
  }

  function drawStory(timeSeconds) {
    const anchors = buildAnchors(timeSeconds);
    const cell = canvasWidth < 700 ? 10 : 12;

    canvasCtx.clearRect(0, 0, canvasWidth, canvasHeight);

    for (let lineY = 0; lineY < canvasHeight; lineY += 72) {
      canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.018)";
      canvasCtx.beginPath();
      canvasCtx.moveTo(0, lineY);
      canvasCtx.lineTo(canvasWidth, lineY);
      canvasCtx.stroke();
    }

    for (let y = 0; y < canvasHeight; y += cell) {
      for (let x = 0; x < canvasWidth; x += cell) {
        let intensity = 0;

        for (const anchor of anchors) {
          const cloudX =
            anchor.side === "right"
              ? canvasWidth * 0.87 + Math.sin(timeSeconds * 0.52 + anchor.phase) * 26
              : canvasWidth * 0.13 + Math.cos(timeSeconds * 0.56 + anchor.phase) * 26;
          const dx = x - cloudX;
          const dy = y - anchor.y;
          intensity += (anchor.radius * anchor.radius) / (dx * dx * 1.08 + dy * dy * 0.84 + anchor.radius * 24);
        }

        const centerBias = 1 - Math.abs(x - canvasWidth * 0.5) / Math.max(canvasWidth * 0.5, 1);
        intensity *= Math.max(0.14, 0.92 - centerBias * 0.62);

        if (intensity < 0.72) continue;

        const shade = Math.min(255, Math.round(128 + intensity * 60));
        const alpha = Math.min(0.5, 0.08 + intensity * 0.08);
        canvasCtx.fillStyle = `rgba(${shade}, ${shade}, ${Math.min(255, shade + 10)}, ${alpha})`;
        canvasCtx.fillRect(x, y, cell - 1, cell - 1);
      }
    }

    canvasCtx.save();
    canvasCtx.lineWidth = 1;
    canvasCtx.setLineDash([8, 12]);
    canvasCtx.lineDashOffset = -timeSeconds * 18;

    anchors.forEach((anchor, index) => {
      const next = anchors[index + 1];
      const spineX = anchor.side === "right" ? canvasWidth * 0.52 : canvasWidth * 0.48;

      canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.14)";
      canvasCtx.beginPath();
      canvasCtx.moveTo(anchor.x, anchor.y);
      canvasCtx.lineTo(spineX, anchor.y);
      canvasCtx.stroke();

      if (!next) return;

      const nextSpineX = next.side === "right" ? canvasWidth * 0.52 : canvasWidth * 0.48;
      canvasCtx.strokeStyle = "rgba(95, 157, 255, 0.22)";
      canvasCtx.beginPath();
      canvasCtx.moveTo(spineX, anchor.y);
      canvasCtx.bezierCurveTo(canvasWidth * 0.5, anchor.y, canvasWidth * 0.5, next.y, nextSpineX, next.y);
      canvasCtx.stroke();
    });

    canvasCtx.restore();

    anchors.forEach((anchor, index) => {
      const glow = canvasCtx.createRadialGradient(anchor.x, anchor.y, 0, anchor.x, anchor.y, 76);
      glow.addColorStop(0, "rgba(95, 157, 255, 0.22)");
      glow.addColorStop(1, "rgba(95, 157, 255, 0)");
      canvasCtx.fillStyle = glow;
      canvasCtx.fillRect(anchor.x - 76, anchor.y - 76, 152, 152);

      const pulseSize = 3 + Math.sin(timeSeconds * 2.2 + index) * 0.8;
      canvasCtx.fillStyle = "rgba(255, 255, 255, 0.82)";
      canvasCtx.fillRect(anchor.x - pulseSize / 2, anchor.y - pulseSize / 2, pulseSize, pulseSize);
    });

    anchors.forEach((anchor, index) => {
      const next = anchors[index + 1];
      if (!next) return;

      const t = (Math.sin(timeSeconds * 0.8 + index * 0.7) + 1) * 0.5;
      const x = canvasWidth * 0.5 + Math.sin(timeSeconds * 0.9 + index) * 16;
      const y = anchor.y + (next.y - anchor.y) * t;
      canvasCtx.fillStyle = "rgba(255, 255, 255, 0.74)";
      canvasCtx.fillRect(x - 1.5, y - 1.5, 3, 3);
    });
  }

  function renderStory(now) {
    if (!lastFrame || now - lastFrame > getMotionFrameInterval(40, 58, 88)) {
      if ((active || prefersReducedMotion.matches) && !isPerformanceLiteScrollActive()) {
        drawStory(now / 1000);
      }
      lastFrame = now;
    }
    window.requestAnimationFrame(renderStory);
  }

  window.addEventListener("resize", resizeStoryCanvas);
  resizeStoryCanvas();
  drawStory(0);
  if (!prefersReducedMotion.matches) {
    window.requestAnimationFrame(renderStory);
  }
}

function initializeChapterPixelStreams() {
  const entries = [...document.querySelectorAll("[data-turn-section]")]
    .map((section, index) => {
      const shell = section.querySelector(".section-turn-shell");
      const canvas = section.querySelector(".section-pixel-canvas");
      const canvasCtx = canvas?.getContext("2d", { alpha: true });
      if (!shell || !canvas || !canvasCtx) return null;
      return {
        section,
        shell,
        canvas,
        canvasCtx,
        index,
        active: true,
        rendered: false,
        width: 0,
        height: 0,
        cell: 0,
        lanes: [],
      };
    })
    .filter(Boolean);

  if (!entries.length) return;

  const pseudoRandom = (seed) => {
    const value = Math.sin(seed * 127.1 + seed * 0.37) * 43758.5453;
    return value - Math.floor(value);
  };

  const bezierPoint = (point0, point1, point2, point3, t) => {
    const inverse = 1 - t;
    const x =
      inverse ** 3 * point0.x +
      3 * inverse ** 2 * t * point1.x +
      3 * inverse * t ** 2 * point2.x +
      t ** 3 * point3.x;
    const y =
      inverse ** 3 * point0.y +
      3 * inverse ** 2 * t * point1.y +
      3 * inverse * t ** 2 * point2.y +
      t ** 3 * point3.y;
    return { x, y };
  };

  const resizeEntry = (entry) => {
    const ratio = getCanvasRatio(1.1, 1);
    entry.width = entry.shell.clientWidth;
    entry.height = entry.shell.clientHeight;
    entry.canvas.width = Math.max(1, Math.floor(entry.width * ratio));
    entry.canvas.height = Math.max(1, Math.floor(entry.height * ratio));
    entry.canvas.style.width = `${entry.width}px`;
    entry.canvas.style.height = `${entry.height}px`;
    entry.canvasCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
    entry.cell = entry.width < 900 ? 4 : 5;

    const direction = entry.index % 2 === 0 ? 1 : -1;
    entry.lanes = Array.from({ length: 3 }, (_, laneIndex) => {
      const laneSeed = entry.index * 10 + laneIndex;
      const laneHeight = entry.height * (0.24 + laneIndex * 0.22);
      const laneSwing = 36 + laneIndex * 12;
      const startX = direction === 1 ? entry.width * 0.06 : entry.width * 0.94;
      const endX = direction === 1 ? entry.width * 0.94 : entry.width * 0.06;
      return {
        phase: pseudoRandom(laneSeed + 3),
        speed: 0.055 + laneIndex * 0.015 + entry.index * 0.006,
        points: [
          { x: startX, y: laneHeight + (pseudoRandom(laneSeed + 7) - 0.5) * 24 },
          {
            x: entry.width * (direction === 1 ? 0.28 : 0.72),
            y: laneHeight - laneSwing + pseudoRandom(laneSeed + 11) * 40,
          },
          {
            x: entry.width * (direction === 1 ? 0.72 : 0.28),
            y: laneHeight + laneSwing - pseudoRandom(laneSeed + 17) * 38,
          },
          { x: endX, y: laneHeight + (pseudoRandom(laneSeed + 23) - 0.5) * 24 },
        ],
      };
    });
  };

  const clearEntry = (entry) => {
    entry.canvasCtx.clearRect(0, 0, entry.width, entry.height);
  };

  const getEntryDetailLevel = (entry) => {
    if (prefersReducedMotion.matches) return 2;
    if (!entry.active) return 0;
    if (!motionProfile.performanceLite) return 2;
    if (isPerformanceLiteScrollActive()) return 0;
    if (entry.section.classList.contains("is-chapter-settled")) return 2;
    if (entry.section.classList.contains("is-chapter-live")) return 1;
    return 0;
  };

  const drawEntry = (entry, timeSeconds, detailLevel = 2) => {
    const { canvasCtx, width, height, cell } = entry;
    const glow = parseFloat(getComputedStyle(entry.section).getPropertyValue("--turn-glow")) || 0.5;
    const assembly = parseFloat(getComputedStyle(entry.section).getPropertyValue("--assembly-progress")) || 0;
    const direction = entry.index % 2 === 0 ? 1 : -1;
    const anchorPointCount = detailLevel === 1 ? 28 : 54;
    const laneStep = detailLevel === 1 ? 0.056 : 0.034;
    const packetCount = detailLevel === 1 ? 2 : 4;
    const trailCount = detailLevel === 1 ? 3 : 5;
    const laneLimit = detailLevel === 1 ? Math.min(2, entry.lanes.length) : entry.lanes.length;

    canvasCtx.clearRect(0, 0, width, height);

    const anchors = [
      {
        x: direction === 1 ? width * 0.11 : width * 0.89,
        y: height * 0.18,
        radius: Math.min(width, height) * 0.18,
      },
      {
        x: direction === 1 ? width * 0.89 : width * 0.11,
        y: height * 0.78,
        radius: Math.min(width, height) * 0.22,
      },
    ];

    anchors.forEach((anchor, anchorIndex) => {
      for (let pointIndex = 0; pointIndex < anchorPointCount; pointIndex += 1) {
        const seed = entry.index * 200 + anchorIndex * 64 + pointIndex;
        const angle = pseudoRandom(seed + 1) * Math.PI * 2 + timeSeconds * (0.08 + anchorIndex * 0.03);
        const radiusFactor = 0.18 + pseudoRandom(seed + 5) * 0.98;
        const drift = Math.sin(timeSeconds * 0.7 + seed) * (5 + (1 - assembly) * 11);
        const spread = 1 + (1 - assembly) * 1.2;
        const scatterX = (pseudoRandom(seed + 21) - 0.5) * (1 - assembly) * 88;
        const scatterY = (pseudoRandom(seed + 27) - 0.5) * (1 - assembly) * 54;
        const pixelX = anchor.x + Math.cos(angle) * anchor.radius * radiusFactor * spread + drift + scatterX;
        const pixelY = anchor.y + Math.sin(angle) * anchor.radius * radiusFactor * spread + scatterY;
        const size = cell * (1 + Math.floor(pseudoRandom(seed + 9) * 2));
        const alpha = (0.04 + pseudoRandom(seed + 13) * 0.14) * (0.24 + glow * 0.58 + assembly * 0.42);
        const shade = 176 + Math.round(pseudoRandom(seed + 17) * 54);
        canvasCtx.fillStyle = `rgba(${shade}, ${shade}, ${Math.min(255, shade + 22)}, ${alpha})`;
        canvasCtx.fillRect(Math.round(pixelX / cell) * cell, Math.round(pixelY / cell) * cell, size, size);
      }
    });

    entry.lanes.forEach((lane, laneIndex) => {
      if (laneIndex >= laneLimit) return;

      for (let t = 0.03; t < 1; t += laneStep) {
        const point = bezierPoint(lane.points[0], lane.points[1], lane.points[2], lane.points[3], t);
        const laneScatterX = (Math.sin(timeSeconds * 0.9 + t * 14 + laneIndex) * (1 - assembly) * 28);
        const laneScatterY = (Math.cos(timeSeconds * 0.7 + t * 12 + laneIndex) * (1 - assembly) * 16);
        const alpha = (0.06 + Math.sin(timeSeconds * 0.8 + t * Math.PI * 2 + laneIndex) * 0.03) * (0.22 + glow * 0.56 + assembly * 0.5);
        canvasCtx.fillStyle = `rgba(128, 176, 255, ${Math.max(0.04, alpha)})`;
        canvasCtx.fillRect(
          Math.round((point.x + laneScatterX) / cell) * cell,
          Math.round((point.y + laneScatterY) / cell) * cell,
          cell + 1,
          cell + 1,
        );
      }

      for (let packetIndex = 0; packetIndex < packetCount; packetIndex += 1) {
        const progress = (timeSeconds * lane.speed + lane.phase + packetIndex * 0.24) % 1;
        for (let trailIndex = 0; trailIndex < trailCount; trailIndex += 1) {
          const trailProgress = (progress - trailIndex * 0.026 + 1) % 1;
          const point = bezierPoint(lane.points[0], lane.points[1], lane.points[2], lane.points[3], trailProgress);
          const trailScatterX = (pseudoRandom(packetIndex * 17 + trailIndex * 23 + laneIndex) - 0.5) * (1 - assembly) * 54;
          const trailScatterY = Math.sin(timeSeconds * 1.1 + trailIndex + laneIndex) * (1 - assembly) * 20;
          const size = cell * (trailIndex === 0 ? 3 : 2);
          const alpha = (0.16 - trailIndex * 0.022 + assembly * 0.16) * (0.32 + glow * 0.5 + assembly * 0.42);
          canvasCtx.fillStyle = `rgba(214, 228, 255, ${Math.max(0.04, alpha)})`;
          canvasCtx.fillRect(
            Math.round((point.x + trailScatterX) / cell) * cell,
            Math.round((point.y + trailScatterY) / cell) * cell,
            size,
            size,
          );
        }
      }
    });
  };

  entries.forEach((entry) => {
    resizeEntry(entry);
  });

  if (!prefersReducedMotion.matches) {
    const entryMap = new Map(entries.map((entry) => [entry.section, entry]));
    const sectionObserver = new IntersectionObserver(
      (sectionEntries) => {
        sectionEntries.forEach((sectionEntry) => {
          const matchedEntry = entryMap.get(sectionEntry.target);
          if (!matchedEntry) return;
          matchedEntry.active = sectionEntry.isIntersecting;
        });
      },
      { threshold: 0.06 },
    );

    entries.forEach((entry) => sectionObserver.observe(entry.section));
  }

  const resizeAll = () => {
    entries.forEach((entry) => resizeEntry(entry));
  };

  let lastFrame = 0;
  const render = (now) => {
    if (!lastFrame || now - lastFrame > getMotionFrameInterval(48, 70, 104)) {
      const timeSeconds = now / 1000;
      entries.forEach((entry) => {
        const detailLevel = getEntryDetailLevel(entry);
        if (!detailLevel) {
          if (entry.rendered) {
            clearEntry(entry);
            entry.rendered = false;
          }
          return;
        }

        drawEntry(entry, timeSeconds, detailLevel);
        entry.rendered = true;
      });
      lastFrame = now;
    }

    window.requestAnimationFrame(render);
  };

  window.addEventListener("resize", resizeAll);
  entries.forEach((entry) => {
    const detailLevel = getEntryDetailLevel(entry);
    if (!detailLevel) return;
    drawEntry(entry, 0, detailLevel);
    entry.rendered = true;
  });
  if (!prefersReducedMotion.matches) {
    window.requestAnimationFrame(render);
  }
}

function initializeCursorRing() {
  if (!cursorRing) return;
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
  if (!finePointer.matches || prefersReducedMotion.matches) return;

  body.classList.add("has-custom-cursor");
  const state = {
    x: window.innerWidth * 0.5,
    y: window.innerHeight * 0.5,
    targetX: window.innerWidth * 0.5,
    targetY: window.innerHeight * 0.5,
  };

  const interactiveSelector = "a, button, [role='button']";

  window.addEventListener("pointermove", (event) => {
    state.targetX = event.clientX;
    state.targetY = event.clientY;
    body.classList.add("cursor-visible");
    const interactive = event.target instanceof Element && event.target.closest(interactiveSelector);
    body.classList.toggle("cursor-active-target", Boolean(interactive));
  });

  window.addEventListener("pointerleave", () => {
    body.classList.remove("cursor-visible");
    body.classList.remove("cursor-active-target");
  });

  const render = () => {
    state.x += (state.targetX - state.x) * 0.22;
    state.y += (state.targetY - state.y) * 0.22;
    cursorRing.style.transform = `translate3d(${state.x}px, ${state.y}px, 0)`;
    window.requestAnimationFrame(render);
  };

  render();
}

const canvas = document.querySelector("#signal-canvas");
const ctx = canvas?.getContext("2d");
let width = 0;
let height = 0;
let points = [];
let tick = 0;
const focus = {
  x: 0,
  y: 0,
  targetX: 0,
  targetY: 0,
};

function resizeCanvas() {
  if (!canvas || !ctx) return;
  const ratio = getCanvasRatio(1.15, 1);
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  points = Array.from({ length: Math.min(motionProfile.performanceLite ? 56 : 96, Math.floor(width / 15)) }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    phase: Math.random() * Math.PI * 2,
  }));
  focus.x = width * 0.5;
  focus.y = height * 0.32;
  focus.targetX = focus.x;
  focus.targetY = focus.y;
}

function initializeCanvasMotion() {
  if (!canvas || !ctx) return;

  window.addEventListener("pointermove", (event) => {
    focus.targetX = event.clientX;
    focus.targetY = Math.min(event.clientY, height * 0.72);
  });

  window.addEventListener("pointerleave", () => {
    focus.targetX = width * 0.5;
    focus.targetY = height * 0.32;
  });

  let lastFrame = 0;
  const draw = (now) => {
    if (!lastFrame || now - lastFrame > getMotionFrameInterval(34, 52, 86)) {
      lastFrame = now;
      if (isPerformanceLiteScrollActive() && window.scrollY > window.innerHeight * 0.45) {
        window.requestAnimationFrame(draw);
        return;
      }

      tick += 0.008;
      focus.x += (focus.targetX - focus.x) * 0.03;
      focus.y += (focus.targetY - focus.y) * 0.03;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, width, height);

      const halo = ctx.createRadialGradient(focus.x, focus.y, 0, focus.x, focus.y, Math.min(width, height) * 0.28);
      halo.addColorStop(0, "rgba(255,255,255,0.08)");
      halo.addColorStop(0.34, "rgba(95,157,255,0.06)");
      halo.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = halo;
      ctx.fillRect(0, 0, width, height);

      for (const point of points) {
        point.x += point.vx;
        point.y += point.vy;
        if (point.x < 0 || point.x > width) point.vx *= -1;
        if (point.y < 0 || point.y > height) point.vy *= -1;
      }

      ctx.lineWidth = 1;
      const linkDistance = motionProfile.performanceLite ? 96 : 120;
      for (let i = 0; i < points.length; i += 1) {
        for (let j = i + 1; j < points.length; j += 1) {
          const pointA = points[i];
          const pointB = points[j];
          const dx = pointA.x - pointB.x;
          const dy = pointA.y - pointB.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < linkDistance) {
            const alpha = (1 - distance / linkDistance) * 0.08;
            ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(pointA.x, pointA.y);
            ctx.lineTo(pointB.x, pointB.y);
            ctx.stroke();
          }
        }
      }

      for (let lineIndex = 0; lineIndex < 5; lineIndex += 1) {
        const lineY = (tick * 180 + lineIndex * (height / 5)) % height;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.018)";
        ctx.beginPath();
        ctx.moveTo(0, lineY);
        ctx.lineTo(width, lineY);
        ctx.stroke();
      }

      for (const point of points) {
        const pulse = 0.12 + Math.sin(tick + point.phase) * 0.05;
        ctx.fillStyle = `rgba(180, 190, 205, ${pulse})`;
        ctx.fillRect(point.x - 1, point.y - 1, 2, 2);
      }
    }

    window.requestAnimationFrame(draw);
  };

  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  draw(0);
}

initializeLivePreview();
initializeRevealAnimations();
initializeActiveNav();
initializeScrollState();
initializeInitialSectionJump();
initializeHeroParallax();
initializeTurnSections();
initializeCanvasMotion();
initializeChapterPixelStreams();
initializeManifestoCanvas();
initializeStoryCanvas();
initializeCursorRing();
