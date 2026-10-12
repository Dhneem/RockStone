// 2-D slice viewer: window/level drag, slice scrolling, overlays, ROI drawing.
import { sliceUrl } from "./api.js?v=2.1.3";

/** Show a data-URL / path image in a canvas *or* an <img> element. */
export function drawUrl(target, src) {
  return new Promise((resolve) => {
    if (!src || !target) { resolve(null); return; }
    const img = new Image();
    img.onload = () => {
      if (typeof target.getContext === "function") {
        const ctx = target.getContext("2d");
        target.width = img.naturalWidth;
        target.height = img.naturalHeight;
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, target.width, target.height);
        ctx.drawImage(img, 0, 0);
      } else {
        target.src = src;   // plain <img>
      }
      resolve(img);
    };
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

export class SliceViewer {
  /**
   * hooks: {
   *   onWindowChange(lo, hi), onWheel(delta, index), onDoubleClick(),
   *   onShapeChange(shape), onDrawClick(pt, kind), onDrawFinish()
   * }
   */
  constructor(canvas, hooks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.hooks = hooks;
    this.state = {
      index: 0, lo: -1000, hi: 1000, invert: false, overlay: "", color: "",
      ready: false,
    };
    this.shape = null;      // {type, x, y, w, h} | {type, cx, cy, rx, ry} | {type, points:[[x,y]]}
    this.drag = null;       // ROI drag state
    this.winDrag = null;    // window/level drag state
    this.measure = null;    // {a:[x,y], b:[x,y], done:bool, dragging:bool}
    this.caliper = null;    // in-progress thickness caliper (same shape as measure)
    this.calipers = [];     // completed thickness calipers across a crack/layer
    this.spacing = null;    // [mm per row, mm per col] - enables mm units on measure
    this.imgW = 512;
    this.imgH = 512;
    this._token = 0;
    this._bind();
  }

  configure(opts) {
    Object.assign(this.state, opts);
    return this.refresh();
  }

  setTool(tool) {
    if (tool !== this.tool) {
      // A half-drawn annotation dies with its tool; completed ones persist
      // (only Clear or toggling the tool off removes those — see the manual).
      const hadPartial = this.caliper || (this.measure && !this.measure.done);
      this.caliper = null;
      if (this.measure && !this.measure.done) this.measure = null;
      if (hadPartial) this._paint();
    }
    this.tool = tool;   // "window" | "rect" | "ellipse" | "polygon" | "measure" | "thickness"
    this.canvas.style.cursor = tool === "rect" || tool === "ellipse" || tool === "polygon" ? "cell" : "crosshair";
  }

  clearMeasure() {
    this.measure = null;
    this._paint();
    this.hooks.onMeasureChange?.(null);
  }

  clearThickness() {
    this.caliper = null;
    this.calipers = [];
    this._paint();
    this.hooks.onThicknessChange?.(null);
  }

  /** Completed thickness calipers as {n, values, px:{mean,min,max,std}, mm|}. */
  thicknessValue() {
    if (!this.calipers.length) return null;
    const values = this.calipers.map(({ a, b }) => {
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const px = Math.hypot(dx, dy);
      const mm = this.spacing ? Math.hypot(dx * this.spacing[1], dy * this.spacing[0]) : null;
      return { px, mm };
    });
    const stats = (arr) => {
      const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
      return { mean, min: Math.min(...arr), max: Math.max(...arr),
               std: Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length) };
    };
    return { n: values.length, values,
             px: stats(values.map((v) => v.px)),
             mm: this.spacing ? stats(values.map((v) => v.mm)) : null };
  }

  /** Current measurement as {a, b, px, mm|null}. */
  measureValue() {
    const m = this.measure;
    if (!m) return null;
    const dx = m.b[0] - m.a[0], dy = m.b[1] - m.a[1];
    const px = Math.hypot(dx, dy);
    const mm = this.spacing ? Math.hypot(dx * this.spacing[1], dy * this.spacing[0]) : null;
    return { a: m.a, b: m.b, px, mm };
  }

  setShape(shape) {
    this.shape = shape;
    this._paint();
  }

  async refresh() {
    const s = this.state;
    const url = sliceUrl({ index: s.index, lo: s.lo, hi: s.hi, invert: s.invert, overlay: s.overlay, color: s.color });
    const token = ++this._token;
    const img = await new Promise((resolve) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = () => resolve(null);
      i.src = url + "&t=" + Date.now();
    });
    if (token !== this._token || !img) return;   // a newer request won
    this.img = img;
    this.imgW = img.naturalWidth;
    this.imgH = img.naturalHeight;
    this.state.ready = true;
    this._paint();
  }

  // ---------------------------------------------------------------- drawing
  _paint() {
    const { canvas, ctx } = this;
    if (this.img) {
      canvas.width = this.imgW;
      canvas.height = this.imgH;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.img, 0, 0);
    }
    const sh = this.shape;
    if (sh) {
      ctx.save();
      ctx.lineWidth = Math.max(1.5, canvas.width / 180);
      ctx.strokeStyle = "#2ed3a2";
      ctx.fillStyle = "rgba(46, 211, 162, .14)";
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      if (sh.type === "rect") {
        ctx.rect(sh.x, sh.y, sh.w, sh.h);
        ctx.fill(); ctx.stroke();
      } else if (sh.type === "ellipse" || sh.type === "circle") {
        ctx.ellipse(sh.cx, sh.cy, Math.abs(sh.rx), Math.abs(sh.ry), 0, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
      } else if (sh.type === "polygon" && sh.points.length) {
        ctx.setLineDash([]);
        sh.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
        ctx.fillStyle = "#2ed3a2";
        sh.points.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fill(); });
      }
      ctx.restore();
    }
    this._paintMeasure();
    this._paintThickness();
  }

  /** Small dark chip with text, clamped inside the canvas. */
  _chip(x, y, text, fg, fs) {
    const { canvas, ctx } = this;
    fs = fs || Math.max(11, Math.round(canvas.width / 38));
    ctx.font = `bold ${fs}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const tw = ctx.measureText(text).width;
    const cx = Math.min(Math.max(x, tw / 2 + 4), canvas.width - tw / 2 - 4);
    const cy = Math.min(Math.max(y, fs), canvas.height - fs);
    ctx.fillStyle = "rgba(8, 12, 20, .72)";
    ctx.fillRect(cx - tw / 2 - 5, cy - fs * 0.72, tw + 10, fs * 1.44);
    ctx.fillStyle = fg;
    ctx.fillText(text, cx, cy);
  }

  _paintMeasure() {
    const m = this.measure;
    if (!m) return;
    const { canvas, ctx } = this;
    const v = this.measureValue();
    ctx.save();
    ctx.setLineDash([]);
    ctx.lineWidth = Math.max(1.5, canvas.width / 180);
    ctx.strokeStyle = "#50c8ff";
    ctx.beginPath();
    ctx.moveTo(m.a[0], m.a[1]);
    ctx.lineTo(m.b[0], m.b[1]);
    ctx.stroke();
    ctx.fillStyle = "#50c8ff";
    for (const p of [m.a, m.b]) { ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 7); ctx.fill(); }
    const label = v.mm != null ? `${v.mm.toFixed(2)} mm` : `${v.px.toFixed(1)} px`;
    this._chip((m.a[0] + m.b[0]) / 2, (m.a[1] + m.b[1]) / 2 - Math.max(11, Math.round(canvas.width / 38)),
               label, "#d9f1ff");
    ctx.restore();
  }

  _paintThickness() {
    if (!this.calipers.length && !this.caliper) return;
    const { canvas, ctx } = this;
    const lw = Math.max(1.5, canvas.width / 180);
    const fs = Math.max(10, Math.round(canvas.width / 44));
    ctx.save();
    ctx.setLineDash([]);
    ctx.lineWidth = lw;
    // completed calipers
    const all = this.caliper ? [...this.calipers, this.caliper] : this.calipers;
    all.forEach((m, i) => {
      const active = i === all.length - 1 && this.caliper;
      const dx = m.b[0] - m.a[0], dy = m.b[1] - m.a[1];
      const px = Math.hypot(dx, dy);
      const mm = this.spacing ? Math.hypot(dx * this.spacing[1], dy * this.spacing[0]) : null;
      ctx.strokeStyle = active ? "#ffb84d" : "rgba(255, 184, 77, .85)";
      ctx.beginPath();
      ctx.moveTo(m.a[0], m.a[1]);
      ctx.lineTo(m.b[0], m.b[1]);
      ctx.stroke();
      // end ticks perpendicular to the caliper (classic thickness glyph)
      const n = Math.hypot(dx, dy) || 1;
      const tx = -dy / n * lw * 2.5, ty = dx / n * lw * 2.5;
      ctx.beginPath();
      ctx.moveTo(m.a[0] - tx, m.a[1] - ty); ctx.lineTo(m.a[0] + tx, m.a[1] + ty);
      ctx.moveTo(m.b[0] - tx, m.b[1] - ty); ctx.lineTo(m.b[0] + tx, m.b[1] + ty);
      ctx.stroke();
      ctx.fillStyle = "#ffb84d";
      for (const p of [m.a, m.b]) { ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, 7); ctx.fill(); }
      this._chip((m.a[0] + m.b[0]) / 2, (m.a[1] + m.b[1]) / 2 - fs,
                 mm != null ? `${mm.toFixed(2)}` : `${px.toFixed(1)}`, "#ffe4bd", fs);
    });
    // mean chip once 2+ calipers exist
    const v = this.thicknessValue();
    if (v && v.n >= 2) {
      const cx = all.reduce((s, m) => s + (m.a[0] + m.b[0]) / 2, 0) / all.length;
      const cy = all.reduce((s, m) => s + (m.a[1] + m.b[1]) / 2, 0) / all.length;
      const label = v.mm ? `mean ${v.mm.mean.toFixed(2)} mm` : `mean ${v.px.mean.toFixed(1)} px`;
      this._chip(cx, cy + fs * 2.2, label, "#ffb84d", fs);
    }
    ctx.restore();
  }

  /** Is (x, y) inside the current shape? */
  _hit(x, y) {
    const sh = this.shape;
    if (!sh) return false;
    if (sh.type === "rect") return x >= sh.x && x <= sh.x + sh.w && y >= sh.y && y <= sh.y + sh.h;
    if (sh.type === "ellipse" || sh.type === "circle") {
      const rx = Math.abs(sh.rx) || 1, ry = Math.abs(sh.ry) || 1;
      return ((x - sh.cx) / rx) ** 2 + ((y - sh.cy) / ry) ** 2 <= 1;
    }
    if (sh.type === "polygon" && sh.points.length >= 3) {
      let inside = false;
      for (let i = 0, j = sh.points.length - 1; i < sh.points.length; j = i++) {
        const [xi, yi] = sh.points[i], [xj, yj] = sh.points[j];
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
      }
      return inside;
    }
    return false;
  }

  /** Index of a polygon vertex within grab distance of (x, y), or -1. */
  _vertexHit(x, y) {
    const sh = this.shape;
    if (!sh || sh.type !== "polygon") return -1;
    const r = Math.max(6, this.canvas.width / 80);
    for (let i = 0; i < sh.points.length; i++) {
      const [px, py] = sh.points[i];
      if ((px - x) ** 2 + (py - y) ** 2 <= r * r) return i;
    }
    return -1;
  }

  _toImage(ev) {
    const r = this.canvas.getBoundingClientRect();
    return [
      (ev.clientX - r.left) / r.width * this.canvas.width,
      (ev.clientY - r.top) / r.height * this.canvas.height,
    ];
  }

  _bind() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (ev) => {
      const [x, y] = this._toImage(ev);
      const tool = this.tool || "window";
      if (tool === "window") {
        this.winDrag = { x: ev.clientX, y: ev.clientY, lo: this.state.lo, hi: this.state.hi };
      } else if (tool === "rect" || tool === "ellipse") {
        if (this._hit(x, y)) {
          const sh = this.shape;
          this.drag = { mode: "move", x0: x, y0: y,
                        orig: { ...sh } };
        } else {
          this.drag = { mode: "draw", type: tool, x0: x, y0: y };
        }
      } else if (tool === "measure" || tool === "thickness") {
        const p = [x, y];
        const cur = tool === "measure" ? this.measure : this.caliper;
        const setCur = (v) => { if (tool === "measure") this.measure = v; else this.caliper = v; };
        if (!cur || cur.done) {
          setCur({ a: p, b: p, done: false, dragging: true });
        } else {
          cur.b = p;
          cur.done = true;
          cur.dragging = false;
          if (tool === "thickness") {
            // each completed caliper joins the set used for the mean
            this.calipers.push({ a: cur.a, b: cur.b });
            this.caliper = null;
            this.hooks.onThicknessChange?.(this.thicknessValue());
          } else {
            this.hooks.onMeasureChange?.(this.measureValue());
          }
        }
        this._paint();
        c.setPointerCapture(ev.pointerId);
        return;
      } else if (tool === "polygon") {
        const vi = this._vertexHit(x, y);
        if (vi >= 0) {
          this.drag = { mode: "vertex", index: vi };
        } else if (this.shape?.type === "polygon" && this.shape.points.length >= 3 && this._hit(x, y)) {
          this.drag = { mode: "move", x0: x, y0: y,
                        orig: this.shape.points.map((p) => p.slice()) };
        } else {
          if (!this.shape || this.shape.type !== "polygon") this.shape = { type: "polygon", points: [] };
          this.shape.points.push([x, y]);
          this._paint();
          this.hooks.onShapeChange?.(this.shape);
          return;
        }
      }
      c.setPointerCapture(ev.pointerId);
    });

    c.addEventListener("pointermove", (ev) => {
      if (this.measure?.dragging || this.caliper?.dragging) {
        const [x, y] = this._toImage(ev);
        (this.measure?.dragging ? this.measure : this.caliper).b = [x, y];
        this._paint();
      } else if (this.winDrag) {
        const dx = ev.clientX - this.winDrag.x;
        const dy = ev.clientY - this.winDrag.y;
        const span = Math.max(50, this.winDrag.hi - this.winDrag.lo);
        const w = Math.min(Math.max(span * Math.exp(dx / 220), 20), 20000);
        const level = this.winDrag.lo + (this.winDrag.hi - this.winDrag.lo) / 2 - dy * (span / 220);
        this.state.lo = level - w / 2;
        this.state.hi = level + w / 2;
        this.hooks.onWindowChange?.(this.state.lo, this.state.hi);
      } else if (this.drag) {
        const [x, y] = this._toImage(ev);
        const d = this.drag;
        if (d.mode === "move") {
          const dx = x - d.x0, dy = y - d.y0;
          const sh = this.shape;
          if (sh.type === "rect") { sh.x = d.orig.x + dx; sh.y = d.orig.y + dy; }
          else if (sh.type === "ellipse" || sh.type === "circle") { sh.cx = d.orig.cx + dx; sh.cy = d.orig.cy + dy; }
          else if (sh.type === "polygon") sh.points = d.orig.map(([px, py]) => [px + dx, py + dy]);
        } else if (d.mode === "vertex") {
          this.shape.points[d.index] = [x, y];
        } else if (d.type === "rect") {
          this.shape = { type: "rect", x: Math.min(x, d.x0), y: Math.min(y, d.y0),
                         w: Math.abs(x - d.x0), h: Math.abs(y - d.y0) };
        } else {
          this.shape = { type: "ellipse", cx: (x + d.x0) / 2, cy: (y + d.y0) / 2,
                         rx: Math.abs(x - d.x0) / 2, ry: Math.abs(y - d.y0) / 2 };
        }
        this._paint();
      } else if (this.tool === "rect" || this.tool === "ellipse" || this.tool === "polygon") {
        // hover feedback: show a move cursor when grabbing an existing shape
        const [hx, hy] = this._toImage(ev);
        this.canvas.style.cursor = (this._hit(hx, hy) || this._vertexHit(hx, hy) >= 0) ? "move" : "cell";
      }
    });

    const end = () => {
      if (this.drag) {
        this.drag = null;
        this.hooks.onShapeChange?.(this.shape);
      }
      const finish = (m) => {
        m.dragging = false;
        // a real drag (not a plain click) completes the measurement
        return Math.hypot(m.b[0] - m.a[0], m.b[1] - m.a[1]) > 2;
      };
      if (this.measure?.dragging) {
        if (finish(this.measure)) {
          this.measure.done = true;
          this.hooks.onMeasureChange?.(this.measureValue());
        }
      } else if (this.caliper?.dragging) {
        if (finish(this.caliper)) {
          this.calipers.push({ a: this.caliper.a, b: this.caliper.b });
          this.caliper = null;
          this.hooks.onThicknessChange?.(this.thicknessValue());
        }
      }
      this.winDrag = null;
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);

    c.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const dir = ev.deltaY > 0 ? 1 : -1;
      this.hooks.onWheel?.(dir);
    }, { passive: false });

    c.addEventListener("dblclick", () => {
      if (this.tool === "polygon" && this.shape?.points?.length >= 3) {
        this.hooks.onDrawFinish?.(this.shape);
      } else if (this.tool !== "measure" && this.tool !== "thickness") {
        this.hooks.onDoubleClick?.();
      }
    });
  }
}
