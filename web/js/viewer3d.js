// WebGL2 ray-marched volume renderer for the CT workbench.
// The brick endpoint streams a downsampled uint8 volume; we upload it as a
// 3-D texture and march it front-to-back with a window/level transfer function.

const VS = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() { v_uv = a_pos; gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 v_uv;
out vec4 frag;

uniform sampler3D u_tex;
uniform vec3 u_camPos, u_forward, u_right, u_up;
uniform float u_tan, u_aspect;
uniform vec2 u_win;      // window mapped into brick [0,1] range
uniform float u_sigma;
uniform int u_steps;     // ray-march samples, scaled to brick resolution

void main() {
  vec3 rd = normalize(u_forward + u_tan * (v_uv.x * u_aspect * u_right + v_uv.y * u_up));
  vec3 inv = 1.0 / rd;
  vec3 t0 = (vec3(-0.5) - u_camPos) * inv;
  vec3 t1 = (vec3( 0.5) - u_camPos) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), tmin.z);
  float tf = min(min(tmax.x, tmax.y), tmax.z);
  tn = max(tn, 0.0);
  if (tf <= tn) { frag = vec4(0.0, 0.0, 0.0, 0.0); return; }

  float dt = (tf - tn) / float(u_steps);
  vec3 acc = vec3(0.0);
  float alpha = 0.0;
  for (int i = 0; i < u_steps; i++) {
    vec3 p = u_camPos + rd * (tn + (float(i) + 0.5) * dt);
    float v = texture(u_tex, p + 0.5).r;
    float x = clamp((v - u_win.x) / max(u_win.y - u_win.x, 1e-5), 0.0, 1.0);
    if (x > 0.004) {
      float a = 1.0 - exp(-u_sigma * x * dt);
      acc += (1.0 - alpha) * a * vec3(x);
      alpha += (1.0 - alpha) * a;
      if (alpha > 0.995) break;
    }
  }
  // subtle vignette so empty regions read as depth, not black
  float glow = 0.02 * max(0.0, 1.0 - length(v_uv) * 0.9);
  frag = vec4(acc + vec3(glow) * (1.0 - alpha), alpha);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error("shader: " + gl.getShaderInfoLog(s));
  }
  return s;
}

export class VolumeRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { alpha: true, antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;

    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error("link: " + gl.getProgramInfoLog(prog));
    }
    this.prog = prog;
    gl.useProgram(prog);

    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this.u = {};
    for (const name of ["u_tex", "u_camPos", "u_forward", "u_right", "u_up", "u_tan",
                        "u_aspect", "u_win", "u_sigma", "u_steps"]) {
      this.u[name] = gl.getUniformLocation(prog, name);
    }

    this.view = { theta: 0.6, phi: 0.35, dist: 2.3 };
    this.window = { lo: 0, hi: 1 };
    this.brickRange = { lo: 0, hi: 1 };
    this.sigma = 45.0;
    this.autoRotate = true;
    this.texture = null;
    this.shape = null;
    this.running = false;
    this._bindControls();
  }

  // ------------------------------------------------------------------ data
  async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`brick HTTP ${res.status}`);
    const shape = (res.headers.get("X-Shape") || "0,0,0").split(",").map(Number);
    const lo = parseFloat(res.headers.get("X-Lo") || "0");
    const hi = parseFloat(res.headers.get("X-Hi") || "1");
    const spacing = (res.headers.get("X-Spacing") || "1,1,1").split(",").map(Number);
    const factor = parseInt(res.headers.get("X-Factor") || "1", 10);
    const buf = new Uint8Array(await res.arrayBuffer());
    const [z, y, x] = shape;
    if (buf.length !== z * y * x) throw new Error(`brick size mismatch ${buf.length} != ${z * y * x}`);

    const gl = this.gl;
    const maxTex = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) || 256;
    if (Math.max(x, y, z) > maxTex) {
      throw new Error(`brick ${x}×${y}×${z} exceeds GPU 3D-texture limit ${maxTex} - pick a lower Resolution`);
    }
    if (this.texture) gl.deleteTexture(this.texture);
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, x, y, z, 0, gl.RED, gl.UNSIGNED_BYTE, buf);
    this.texture = tex;
    this.shape = { shape, spacing, factor, lo, hi };
    this.brickRange = { lo, hi };
    if (this.window.hi <= this.window.lo) this.window = { lo, hi };
    this.draw();
    return this.shape;
  }

  setWindow(lo, hi) {
    this.window = { lo, hi };
    this.draw();
  }

  reset() {
    this.view = { theta: 0.6, phi: 0.35, dist: 2.3 };
    this.draw();
  }

  // ---------------------------------------------------------------- render
  draw() {
    const gl = this.gl;
    const { canvas } = this;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr) || canvas.width;
    const h = Math.round(canvas.clientHeight * dpr) || canvas.height;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.texture) return;

    const { theta, phi, dist } = this.view;
    const eye = [
      dist * Math.cos(phi) * Math.sin(theta),
      dist * Math.sin(phi),
      dist * Math.cos(phi) * Math.cos(theta),
    ];
    const norm = (v) => { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const fwd = norm([-eye[0], -eye[1], -eye[2]]);
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);

    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.texture);
    gl.uniform1i(this.u.u_tex, 0);
    gl.uniform3fv(this.u.u_camPos, eye);
    gl.uniform3fv(this.u.u_forward, fwd);
    gl.uniform3fv(this.u.u_right, right);
    gl.uniform3fv(this.u.u_up, up);
    gl.uniform1f(this.u.u_tan, Math.tan((45 * Math.PI / 180) / 2));
    gl.uniform1f(this.u.u_aspect, w / h);

    const span = Math.max(this.brickRange.hi - this.brickRange.lo, 1e-6);
    const w0 = (this.window.lo - this.brickRange.lo) / span;
    const w1 = (this.window.hi - this.brickRange.lo) / span;
    gl.uniform2f(this.u.u_win, w0, w1);
    gl.uniform1f(this.u.u_sigma, this.sigma);
    // scale ray-march samples with the brick's resolution (~1.25 samples/voxel
    // along the diagonal) so higher Resolution settings actually resolve detail
    const brickMax = this.shape ? Math.max(...this.shape.shape) : 192;
    gl.uniform1i(this.u.u_steps, Math.min(768, Math.max(220, Math.round(brickMax * 1.25))));

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  start() {
    if (this.running) return;
    this.running = true;
    let last = 0;
    const loop = (t) => {
      if (!this.running) return;
      // auto-rotate at ~10 fps: ray-marching is too heavy for every frame
      if (this.autoRotate && t - last > 100 && !document.hidden) {
        last = t;
        this.view.theta += 0.02;
        this.draw();
      }
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
  }

  _bindControls() {
    const c = this.canvas;
    let last = null;
    c.addEventListener("pointerdown", (ev) => {
      last = { x: ev.clientX, y: ev.clientY };
      c.setPointerCapture(ev.pointerId);
    });
    c.addEventListener("pointermove", (ev) => {
      if (!last) return;
      this.view.theta -= (ev.clientX - last.x) * 0.008;
      this.view.phi = Math.min(1.45, Math.max(-1.45, this.view.phi + (ev.clientY - last.y) * 0.008));
      last = { x: ev.clientX, y: ev.clientY };
      this.draw();
    });
    const up = () => { last = null; };
    c.addEventListener("pointerup", up);
    c.addEventListener("pointercancel", up);
    c.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      this.view.dist = Math.min(6, Math.max(1.15, this.view.dist * (ev.deltaY > 0 ? 1.08 : 0.93)));
      this.draw();
    }, { passive: false });
  }
}
