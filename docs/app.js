"use strict";

const TAU = Math.PI * 2;
const COLORS = ["#2aa9bd", "#3832d0", "#ce2c76", "#d13a5a"];
const mfCache = new Map();

function rng(seed) {
  return function next() {
    seed |= 0;
    seed = seed + 0x6d2b79f5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function normal(random) {
  const u = Math.max(random(), 1e-12);
  const v = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
}

function wrapPi(x) {
  return ((x + Math.PI) % TAU + TAU) % TAU - Math.PI;
}

function webColor(i, n) {
  const hue = 190 + 155 * i / Math.max(1, n - 1);
  return `hsl(${hue},64%,48%)`;
}

function gaussian(w, sigma) {
  return Math.exp(-(w * w) / (2 * sigma * sigma)) / (Math.sqrt(TAU) * sigma);
}

function criticalK(sigma) {
  return 2 * Math.sqrt(2 / Math.PI) * sigma;
}

function selfConsistency(r, k, sigma) {
  const n = 240;
  let sum = 0;
  for (let i = 0; i <= n; i += 1) {
    const x = -1 + 2 * i / n;
    const weight = i === 0 || i === n ? 0.5 : 1;
    sum += weight * Math.sqrt(Math.max(0, 1 - x * x)) * gaussian(k * r * x, sigma);
  }
  return r - k * r * sum * (2 / n);
}

function meanFieldAt(k, sigma) {
  if (k <= criticalK(sigma)) return 0;
  let lo = 1e-5;
  let hi = 0.999999;
  let flo = selfConsistency(lo, k, sigma);
  for (let i = 0; i < 38; i += 1) {
    const mid = (lo + hi) / 2;
    const fm = selfConsistency(mid, k, sigma);
    if (flo * fm <= 0) hi = mid;
    else {
      lo = mid;
      flo = fm;
    }
  }
  return (lo + hi) / 2;
}

function meanFieldCurve(kMax, sigma, points = 180) {
  const key = `${kMax}:${sigma}:${points}`;
  if (mfCache.has(key)) return mfCache.get(key);
  const curve = Array.from({ length: points }, (_, i) => {
    const k = kMax * i / (points - 1);
    return { k, r: meanFieldAt(k, sigma) };
  });
  mfCache.set(key, curve);
  return curve;
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(260, Math.round(rect.width * dpr));
  const h = Math.max(180, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return { w, h };
}

function clear(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
}

function axes(ctx, box, xLabel, yLabel) {
  ctx.strokeStyle = "#d8e0e8";
  ctx.lineWidth = 1;
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.fillStyle = "#17202a";
  ctx.font = "13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, box.x + box.w / 2, box.y + box.h + 32);
  ctx.save();
  ctx.translate(box.x - 38, box.y + box.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawLine(ctx, data, box, xMax, color, width = 3) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  data.forEach((p, idx) => {
    const x = box.x + box.w * p.k / xMax;
    const y = box.y + box.h * (1 - p.r / 1.02);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

class LiveDemo {
  constructor() {
    this.phase = document.querySelector("#phase-canvas");
    this.diagram = document.querySelector("#diagram-canvas");
    this.raster = document.querySelector("#raster-canvas");
    this.nInput = document.querySelector("#live-n");
    this.sigmaInput = document.querySelector("#live-sigma");
    this.kInput = document.querySelector("#live-k");
    this.speedInput = document.querySelector("#live-speed");
    this.paused = false;
    this.seed = 1;
    this.dt = 0.05;
    this.bind();
    this.reset(true);
  }

  bind() {
    const update = () => {
      document.querySelector("#live-n-value").textContent = this.nInput.value;
      document.querySelector("#live-sigma-value").textContent = Number(this.sigmaInput.value).toFixed(2);
      document.querySelector("#live-k-value").textContent = Number(this.kInput.value).toFixed(2);
      document.querySelector("#live-speed-value").textContent = `${Number(this.speedInput.value).toFixed(1)}x`;
      this.reset(true);
    };
    [this.nInput, this.sigmaInput, this.kInput, this.speedInput].forEach(input => input.addEventListener("input", update));
    document.querySelector("#live-reset").addEventListener("click", () => this.reset(true));
    document.querySelector("#live-pause").addEventListener("click", event => {
      this.paused = !this.paused;
      event.currentTarget.textContent = this.paused ? "Resume" : "Pause";
    });
  }

  reset(newOmega) {
    this.n = Number(this.nInput.value);
    this.sigma = Number(this.sigmaInput.value);
    this.k = Number(this.kInput.value);
    this.speed = Number(this.speedInput.value);
    if (newOmega || !this.omega || this.omega.length !== this.n) {
      const random = rng(this.seed++);
      this.omega = Array.from({ length: this.n }, () => 1 + this.sigma * normal(random)).sort((a, b) => a - b);
    }
    this.theta = Array.from({ length: this.n }, () => Math.PI / 2);
    this.t = 0;
    this.history = [];
    this.spikes = [];
    this.r = 0;
    this.mf = meanFieldCurve(2, this.sigma, 200);
    this.draw();
  }

  step() {
    const z = this.theta.reduce((acc, th) => {
      acc.re += Math.cos(th);
      acc.im += Math.sin(th);
      return acc;
    }, { re: 0, im: 0 });
    z.re /= this.n;
    z.im /= this.n;
    const r = Math.hypot(z.re, z.im);
    const psi = Math.atan2(z.im, z.re);
    const next = new Array(this.n);
    for (let i = 0; i < this.n; i += 1) {
      const old = this.theta[i];
      const th = old + this.dt * (this.omega[i] + this.k * r * Math.sin(psi - old));
      const a = wrapPi(old - Math.PI / 2);
      const b = wrapPi(th - Math.PI / 2);
      if ((a < 0 && b >= 0 || a > 0 && b <= 0) && Math.abs(b - a) < Math.PI) this.spikes.push({ t: this.t, i });
      next[i] = th;
    }
    this.theta = next;
    this.t += this.dt;
    this.history.push({ t: this.t, r });
    this.history = this.history.filter(p => p.t > this.t - 25);
    this.spikes = this.spikes.filter(p => p.t > this.t - 120);
    this.r = this.history.reduce((sum, p) => sum + p.r, 0) / Math.max(1, this.history.length);
  }

  tick() {
    if (!this.paused) {
      const steps = Math.max(1, Math.round(10 * this.speed));
      for (let i = 0; i < steps; i += 1) this.step();
      this.draw();
    }
    requestAnimationFrame(() => this.tick());
  }

  draw() {
    this.drawPhase();
    this.drawDiagram();
    this.drawRaster();
    document.querySelector("#live-r").textContent = this.r.toFixed(3);
  }

  drawPhase() {
    const ctx = this.phase.getContext("2d");
    const { w, h } = fitCanvas(this.phase);
    clear(ctx, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.34;
    ctx.strokeStyle = "#17202a";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, TAU);
    ctx.stroke();
    ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx, cy - radius * 1.12);
    ctx.stroke();
    ctx.setLineDash([]);
    let re = 0;
    let im = 0;
    this.theta.forEach((th, i) => {
      re += Math.cos(th);
      im += Math.sin(th);
      ctx.fillStyle = webColor(i, this.n);
      ctx.beginPath();
      ctx.arc(cx + radius * Math.cos(th), cy - radius * Math.sin(th), 5.2, 0, TAU);
      ctx.fill();
    });
    re /= this.n;
    im /= this.n;
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + radius * re, cy - radius * im);
    ctx.stroke();
  }

  drawDiagram() {
    const ctx = this.diagram.getContext("2d");
    const { w, h } = fitCanvas(this.diagram);
    clear(ctx, w, h);
    const box = { x: 54, y: 24, w: w - 80, h: h - 72 };
    axes(ctx, box, "K", "R");
    drawLine(ctx, this.mf, box, 2, "rgba(80,80,80,0.8)", 3);
    const x = box.x + box.w * this.k / 2;
    const y = box.y + box.h * (1 - this.r / 1.02);
    ctx.strokeStyle = "#000";
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(x, box.y);
    ctx.lineTo(x, box.y + box.h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS[0];
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, TAU);
    ctx.fill();
    ctx.stroke();
  }

  drawRaster() {
    const ctx = this.raster.getContext("2d");
    const { w, h } = fitCanvas(this.raster);
    clear(ctx, w, h);
    const box = { x: 50, y: 20, w: w - 70, h: h - 58 };
    axes(ctx, box, "t", "i");
    const t0 = Math.max(0, this.t - 120);
    this.spikes.forEach(sp => {
      const x = box.x + box.w * (sp.t - t0) / 120;
      const y = box.y + box.h * (1 - sp.i / Math.max(1, this.n - 1));
      ctx.fillStyle = webColor(sp.i, this.n);
      ctx.beginPath();
      ctx.arc(x, y, 2.8, 0, TAU);
      ctx.fill();
    });
  }
}

class TransitionDemo {
  constructor() {
    this.canvas = document.querySelector("#transition-canvas");
    this.sigmaInput = document.querySelector("#sweep-sigma");
    this.kMaxInput = document.querySelector("#sweep-kmax");
    this.pointsInput = document.querySelector("#sweep-points");
    this.cancel = false;
    this.results = [];
    this.bind();
    this.draw();
  }

  bind() {
    const update = () => {
      document.querySelector("#sweep-sigma-value").textContent = Number(this.sigmaInput.value).toFixed(2);
      document.querySelector("#sweep-kmax-value").textContent = Number(this.kMaxInput.value).toFixed(2);
      document.querySelector("#sweep-points-value").textContent = this.pointsInput.value;
      this.results = [];
      this.draw();
    };
    [this.sigmaInput, this.kMaxInput, this.pointsInput].forEach(input => input.addEventListener("input", update));
    document.querySelectorAll(".sweep-n").forEach(input => input.addEventListener("change", update));
    document.querySelector("#run-sweep").addEventListener("click", () => this.run());
    document.querySelector("#cancel-sweep").addEventListener("click", () => {
      this.cancel = true;
    });
  }

  ns() {
    return Array.from(document.querySelectorAll(".sweep-n")).filter(x => x.checked).map(x => Number(x.value));
  }

  async run() {
    this.cancel = false;
    this.results = [];
    const sigma = Number(this.sigmaInput.value);
    const kMax = Number(this.kMaxInput.value);
    const points = Number(this.pointsInput.value);
    const ns = this.ns();
    const kList = Array.from({ length: points }, (_, i) => kMax * i / (points - 1));
    let done = 0;
    const total = ns.length * kList.length;
    document.querySelector("#run-sweep").disabled = true;
    document.querySelector("#cancel-sweep").disabled = false;
    for (let ni = 0; ni < ns.length; ni += 1) {
      const n = ns[ni];
      const random = rng(100 + n);
      const omega = Array.from({ length: n }, () => 1 + sigma * normal(random));
      const series = { n, color: COLORS[ni % COLORS.length], points: [] };
      this.results.push(series);
      for (const k of kList) {
        if (this.cancel) break;
        const out = this.simulate(n, omega, k, random);
        series.points.push({ k, r: out.mean, err: out.std });
        done += 1;
        document.querySelector("#sweep-progress").style.width = `${100 * done / total}%`;
        document.querySelector("#sweep-status").textContent = `Running N=${n}, K=${k.toFixed(2)}`;
        this.draw();
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    document.querySelector("#run-sweep").disabled = false;
    document.querySelector("#cancel-sweep").disabled = true;
    document.querySelector("#sweep-status").textContent = this.cancel ? "Cancelled." : "Complete.";
  }

  simulate(n, omega, k, random) {
    const dt = 0.05;
    const total = n >= 10000 ? 700 : 1200;
    const burn = Math.floor(total / 2);
    let theta = Array.from({ length: n }, () => TAU * random());
    let sum = 0;
    let sumSq = 0;
    let count = 0;
    for (let step = 0; step < total; step += 1) {
      let re = 0;
      let im = 0;
      for (const th of theta) {
        re += Math.cos(th);
        im += Math.sin(th);
      }
      re /= n;
      im /= n;
      const r = Math.hypot(re, im);
      const psi = Math.atan2(im, re);
      theta = theta.map((th, i) => th + dt * (omega[i] + k * r * Math.sin(psi - th)));
      if (step > burn) {
        sum += r;
        sumSq += r * r;
        count += 1;
      }
    }
    const mean = sum / count;
    return { mean, std: Math.sqrt(Math.max(0, sumSq / count - mean * mean)) };
  }

  draw() {
    const ctx = this.canvas.getContext("2d");
    const { w, h } = fitCanvas(this.canvas);
    clear(ctx, w, h);
    const kMax = Number(this.kMaxInput.value);
    const sigma = Number(this.sigmaInput.value);
    const box = { x: 56, y: 28, w: w - 86, h: h - 80 };
    axes(ctx, box, "K", "R");
    drawLine(ctx, meanFieldCurve(kMax, sigma, 200), box, kMax, "rgba(80,80,80,0.8)", 4);
    this.results.forEach(series => {
      ctx.fillStyle = series.color;
      ctx.strokeStyle = "#17202a";
      series.points.forEach(p => {
        const x = box.x + box.w * p.k / kMax;
        const y = box.y + box.h * (1 - p.r / 1.02);
        const dy = box.h * p.err / 1.02;
        ctx.strokeStyle = series.color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y - dy);
        ctx.lineTo(x, y + dy);
        ctx.stroke();
        ctx.fillStyle = series.color;
        ctx.strokeStyle = "#17202a";
        ctx.beginPath();
        ctx.arc(x, y, 4.8, 0, TAU);
        ctx.fill();
        ctx.stroke();
      });
    });
  }
}

document.querySelectorAll(".nav-link").forEach(link => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach(item => item.classList.remove("active"));
    link.classList.add("active");
  });
});

const navTargets = [...document.querySelectorAll(".nav-link")]
  .map(link => ({ link, section: document.querySelector(link.getAttribute("href")) }))
  .filter(item => item.section);

function syncActiveNav() {
  const anchor = window.scrollY + Math.min(window.innerHeight * 0.32, 280);
  let current = navTargets[0];
  navTargets.forEach(item => {
    if (item.section.offsetTop <= anchor) current = item;
  });
  document.querySelectorAll(".nav-link").forEach(item => item.classList.remove("active"));
  if (current) current.link.classList.add("active");
}

window.addEventListener("scroll", syncActiveNav, { passive: true });

document.querySelectorAll(".code-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".code-tab").forEach(item => item.classList.remove("active"));
    document.querySelectorAll(".code-block").forEach(item => item.classList.remove("active"));
    tab.classList.add("active");
    document.querySelector(`[data-code-panel="${tab.dataset.code}"]`).classList.add("active");
  });
});

const liveDemo = new LiveDemo();
const transitionDemo = new TransitionDemo();
window.addEventListener("resize", () => {
  syncActiveNav();
  liveDemo.draw();
  transitionDemo.draw();
});
syncActiveNav();
liveDemo.tick();
