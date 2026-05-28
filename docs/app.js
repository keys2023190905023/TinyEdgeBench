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
operatorFamilies.forEach((name) => {
  const chip = document.createElement("span");
  chip.className = "operator-chip";
  chip.textContent = name;
  cloud.appendChild(chip);
});

const canvas = document.querySelector("#signal-canvas");
const ctx = canvas.getContext("2d");
let width = 0;
let height = 0;
let points = [];

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
  }));
}

function draw() {
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
        const alpha = (1 - distance / 118) * 0.18;
        ctx.strokeStyle = `rgba(98, 240, 232, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  }

  for (const p of points) {
    ctx.fillStyle = "rgba(156, 241, 109, 0.35)";
    ctx.fillRect(p.x - 1, p.y - 1, 2, 2);
  }

  window.requestAnimationFrame(draw);
}

window.addEventListener("resize", resize);
resize();
draw();
