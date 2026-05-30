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
    error: 0.0000,
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
    error: 0.0000,
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

let liveIndex = 0;
updateLiveRun(liveRuns[liveIndex]);
window.setInterval(() => {
  liveIndex = (liveIndex + 1) % liveRuns.length;
  updateLiveRun(liveRuns[liveIndex]);
}, 2400);

const canvas = document.querySelector("#signal-canvas");
const ctx = canvas.getContext("2d");
let width = 0;
let height = 0;
let points = [];
let tick = 0;

function resize() {
  const ratio = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  points = Array.from({ length: Math.min(90, Math.floor(width / 16)) }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * 0.35,
    vy: (Math.random() - 0.5) * 0.35,
    phase: Math.random() * Math.PI * 2,
  }));
}

function draw() {
  tick += 0.012;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#05080d";
  ctx.fillRect(0, 0, width, height);

  for (const p of points) {
    p.x += p.vx;
    p.y += p.vy;
    if (p.x < 0 || p.x > width) p.vx *= -1;
    if (p.y < 0 || p.y > height) p.vy *= -1;
  }

  ctx.lineWidth = 1;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      const a = points[i];
      const b = points[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < 118) {
        const alpha = (1 - distance / 118) * 0.2;
        ctx.strokeStyle = `rgba(98, 240, 232, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  }

  for (const p of points) {
    const pulse = 0.28 + Math.sin(tick + p.phase) * 0.14;
    ctx.fillStyle = `rgba(156, 241, 109, ${pulse})`;
    ctx.fillRect(p.x - 1.2, p.y - 1.2, 2.4, 2.4);
  }

  window.requestAnimationFrame(draw);
}

window.addEventListener("resize", resize);
resize();
draw();
