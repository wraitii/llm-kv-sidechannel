"""Interactive hover-to-inspect attention viewer for one chess example.

Renders a self-contained HTML page (no external assets) with the full
per-layer, per-head attention tensor embedded as JSON. Hovering a token
highlights what it attends to (outgoing) or what attends to it (incoming)
and lists the strongest links.

Usage:
    uv run llmpr-attention-html --config configs/chess-move-bpe512-fen512-causal-20m.json --index 3
    uv run llmpr-attention-html ... --serve   # serve at http://localhost:8765
"""
from __future__ import annotations

import argparse
import functools
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import mlx.core as mx
import numpy as np

from .tokenizer import PairTokenizer, load_tokenizer
from .train import dtype_for, latest_checkpoint, load_model_checkpoint
from .model import model_from_config
from .inspect_attention import build_sequence, target_token_labels

_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>attention inspector</title>
<style>
body{font:13px/1.5 ui-monospace,Menlo,monospace;margin:16px;background:#fafafa;color:#222}
#controls{margin-bottom:10px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0 0 8px}
.meta{color:#666;margin-bottom:10px}
.section{margin:14px 0}
.tok{display:inline-block;padding:1px 3px;margin:1px;border-radius:3px;cursor:default;
     background:#eee;white-space:pre}
.tok.src{background:#e8eef7}.tok.tgt{background:#e9f3e9}
.tok.struct{background:#f3e8f7}
.tok:hover{outline:1px solid #888}
.tok.top{outline:1px solid #d33}
#panel{position:fixed;right:16px;top:16px;width:340px;max-height:90vh;overflow:auto;
       background:#fff;border:1px solid #ccc;border-radius:6px;padding:10px;
       box-shadow:0 2px 8px rgba(0,0,0,.15)}
#panel h2{font-size:13px;margin:0 0 6px}
.row{display:flex;gap:6px;align-items:center;margin:1px 0}
.row .lab{width:150px;overflow:hidden;text-overflow:ellipsis;white-space:pre}
.row .bar{height:10px;background:#4a7ec2;border-radius:2px}
.row .val{width:52px;text-align:right;color:#666}
.hint{color:#888;font-size:11px}
select,input[type=range]{vertical-align:middle}
.legend{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
</style></head><body>
<h1>attention inspector</h1>
<div class="meta">__META__</div>
<div id="controls">
  <label>layer <input type="range" id="layer" min="0" max="__LMAX__" value="__LMAX__">
        <span id="layerv"></span></label>
  <label>head <select id="head"></select></label>
  <label><input type="checkbox" id="incoming"> incoming (who attends to me)</label>
  <label><input type="checkbox" id="top5" checked> outline top-5</label>
</div>
<div class="section"><b>source</b> <span class="hint">(moves; hover to inspect)</span><br>
<span id="src"></span></div>
<div class="section"><b>target</b> <span class="hint">(FEN with squares)</span><br>
<span id="tgt"></span></div>
<div id="panel"><h2 id="ptitle">hover a token</h2><div id="rows">
<span class="hint">hover any token to see its attention links; click to pin.</span>
</div></div>
<script>
const DATA = __DATA__;
const T = DATA.labels.length, NS = DATA.n_source, SEP = NS + 1;
const srcEl = document.getElementById('src'), tgtEl = document.getElementById('tgt');
const layerEl = document.getElementById('layer'), headEl = document.getElementById('head');
const incomingEl = document.getElementById('incoming'), top5El = document.getElementById('top5');
let pinned = -1;

function spans() {
  srcEl.innerHTML = ''; tgtEl.innerHTML = '';
  for (let i = 0; i < T; i++) {
    const s = document.createElement('span');
    s.className = 'tok ' + (i <= SEP ? 'struct' : i <= NS ? 'src' : 'tgt');
    s.textContent = DATA.labels[i];
    s.dataset.i = i;
    s.title = 'pos ' + i + (DATA.groups[i] >= 0 ? ' (move ' + DATA.groups[i] + ')' : '');
    s.addEventListener('mouseenter', () => show(i));
    s.addEventListener('mouseleave', () => { if (pinned < 0) clear(); });
    s.addEventListener('click', () => { pinned = (pinned === i ? -1 : i); show(i); });
    (i <= NS ? srcEl : tgtEl).appendChild(s);
  }
}
function probs() {
  const layer = +layerEl.value;
  document.getElementById('layerv').textContent = layer;
  const head = headEl.value;
  const A = DATA.attention[layer];
  if (head === 'mean') {
    const out = new Float32Array(T * T);
    for (let h = 0; h < A.length; h++) {
      const flat = A[h].flat();
      for (let k = 0; k < out.length; k++) out[k] += flat[k];
    }
    for (let k = 0; k < out.length; k++) out[k] /= A.length;
    return out;
  }
  return A[+head].flat();
}
function show(i) {
  const p = probs();
  const inc = incomingEl.checked;
  document.getElementById('ptitle').textContent =
    (inc ? 'incoming: ' : 'outgoing: ') + DATA.labels[i] +
    (DATA.groups[i] >= 0 ? ' (move ' + DATA.groups[i] + ')' : '') + (pinned === i ? ' [pinned]' : '');
  const rows = [];
  for (let j = 0; j < T; j++) {
    const v = inc ? p[j * T + i] : p[i * T + j];
    if (v > 1e-4) rows.push([j, v]);
  }
  rows.sort((a, b) => b[1] - a[1]);
  const max = rows.length ? rows[0][1] : 1;
  const html = rows.slice(0, 20).map(([j, v]) =>
    '<div class="row"><span class="lab">' + DATA.labels[j] +
    '</span><span class="bar" style="width:' + Math.round(140 * v / max) + 'px' +
    '"></span><span class="val">' + (100 * v).toFixed(1) + '%</span></div>').join('');
  document.getElementById('rows').innerHTML =
    html || '<span class="hint">nothing (masked out)</span>';
  // background highlight by attention, top-5 outline
  const top = rows.slice(0, 5).map(r => r[0]);
  document.querySelectorAll('.tok').forEach(el => {
    const j = +el.dataset.i;
    const v = inc ? p[j * T + i] : p[i * T + j];
    const scale = Math.min(1, v * 14);
    el.style.backgroundColor = '';
    el.style.opacity = 1;
    if (v > 1e-3) {
      el.style.backgroundColor = inc ? 'rgba(217,83,79,' + scale.toFixed(3) + ')'
                                     : 'rgba(66,139,202,' + scale.toFixed(3) + ')';
    }
    el.classList.toggle('top', top5El.checked && top.includes(j));
  });
}
function clear() {
  document.getElementById('ptitle').textContent = 'hover a token';
  document.getElementById('rows').innerHTML =
    '<span class="hint">hover any token to see its attention links; click to pin.</span>';
  document.querySelectorAll('.tok').forEach(el => {
    el.style.backgroundColor = ''; el.classList.remove('top');
  });
}
layerEl.addEventListener('input', () => { pinned >= 0 ? show(pinned) : clear(); });
headEl.addEventListener('change', () => { pinned >= 0 ? show(pinned) : clear(); });
incomingEl.addEventListener('change', () => { pinned >= 0 ? show(pinned) : clear(); });
top5El.addEventListener('change', () => { pinned >= 0 ? show(pinned) : clear(); });
(function init() {
  const nl = DATA.attention.length, nh = DATA.attention[0].length;
  layerEl.max = nl - 1; layerEl.value = nl - 1;
  headEl.innerHTML = '<option value="mean">mean</option>' +
    Array.from({length: nh}, (_, h) => '<option value="' + h + '">head ' + h + '</option>').join('');
  document.getElementById('layerv').textContent = layerEl.value;
  spans();
})();
</script></body></html>
"""


def gather(config: Path, checkpoint: str, split: str, index: int):
    cfg = json.loads(config.read_text())
    tokenizer = PairTokenizer(load_tokenizer(cfg["source_tokenizer"]),
                              load_tokenizer(cfg["target_tokenizer"]),
                              cfg.get("pause_token", False), cfg.get("pause_tokens"),
                              cfg.get("distinct_pause_tokens", False),
                              cfg.get("causal_pause", False))
    model = model_from_config(tokenizer.source.vocab_size, tokenizer.target.vocab_size,
                              cfg, dtype_for(cfg["dtype"]))
    run_dir = Path(cfg["run_dir"])
    path = latest_checkpoint(run_dir) if checkpoint == "auto" else Path(checkpoint)
    state = load_model_checkpoint(path, model)

    with (Path(cfg["data_dir"]) / f"{split}.jsonl").open() as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            if row_index == index:
                row = json.loads(line)
                break
        else:
            raise SystemExit(f"index {index} not found in {handle.name}")

    sequence, valid, prefix_lengths, labels, groups = build_sequence(
        tokenizer, row, cfg)
    tokens = mx.array(sequence)
    sliding = cfg.get("sliding_window") if model.attention_mode == "causal" else None
    _, maps = model.attention_maps(
        tokens, mx.array(valid), mx.array(prefix_lengths), sliding_window=sliding)
    attention = np.stack([np.asarray(p, dtype=np.float32)[0] for p in maps])
    from .inspect_attention import source_pieces
    n_source = len(source_pieces(tokenizer, row["asm"])[:cfg["max_source_tokens"]])
    return attention, labels, groups, n_source, row, state, path, cfg


def render(attention: np.ndarray, labels: list, groups: list, n_source: int,
           row: dict, state: dict, checkpoint: Path, cfg: dict) -> str:
    data = {
        "labels": labels,
        "groups": [-2 if g is None else g for g in groups],
        "n_source": n_source,
        # round to 3 decimals to keep the embedded JSON small
        "attention": np.round(attention, 3).tolist(),
    }
    meta = (f"{row.get('example_id', '?')} · plies={row.get('plies', '?')} · "
            f"fen={row['code']}<br>checkpoint={checkpoint.name} "
            f"(step {state.get('step')}) · layers={attention.shape[0]} · "
            f"heads={attention.shape[1]} · mode={cfg.get('attention_mode', 'prefix')}")
    html = (_TEMPLATE
            .replace("__META__", meta)
            .replace("__LMAX__", str(attention.shape[0] - 1))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    return html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive HTML attention viewer for one chess example")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--serve", action="store_true",
                        help="serve the output directory at localhost:8765")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    attention, labels, groups, n_source, row, state, checkpoint, cfg = gather(
        args.config, args.checkpoint, args.split, args.index)
    out = args.out or (Path(cfg["run_dir"]) / "attention"
                       / f"{args.split}-{args.index}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(attention, labels, groups, n_source, row, state,
                          checkpoint, cfg))
    print(f"wrote {out}")
    if not args.serve:
        print("open it directly in a browser, or rerun with --serve")
        return
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(out.parent))
    url = f"http://localhost:{args.port}/{out.name}"
    print(f"serving {url} (ctrl-C to stop)")
    HTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
