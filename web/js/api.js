// Small API client for the CT workbench backend.
export async function get(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await errText(res));
  return res.json();
}

export async function post(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await errText(res));
  return res.json();
}

export async function del(url) {
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) throw new Error(await errText(res));
  return res.json();
}

export async function binary(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await errText(res));
  return { buf: await res.arrayBuffer(), headers: res.headers };
}

export async function upload(url, files) {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await errText(res));
  return res.json();
}

async function errText(res) {
  try {
    const j = await res.json();
    return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || j);
  } catch {
    return `HTTP ${res.status}`;
  }
}

// Build a slice image URL (server renders window/level + overlays).
export function sliceUrl({ index, lo, hi, invert = false, overlay = "", color = "" } = {}) {
  const p = new URLSearchParams();
  if (index !== undefined && index !== null) p.set("index", index);
  if (lo !== undefined && lo !== null) p.set("lo", lo);
  if (hi !== undefined && hi !== null) p.set("hi", hi);
  if (invert) p.set("invert", "1");
  if (overlay) p.set("overlay", overlay);
  if (color) p.set("color", color);
  return `/api/slice.png?${p.toString()}`;
}
