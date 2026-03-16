#!/usr/bin/env python3
"""Generate a self-contained HTML demo page from CPU benchmark JSON results.

Each JSON must have been produced with --save_samples (and optionally --audio_dir).
Audio files are embedded as base64 so the HTML page is fully self-contained.

Usage:
    python make_demo_html.py \
        --configs \
            "Original FP32:cpu_fp32_original.json" \
            "mid3-22 INT8:cpu_int8_v3-td50-mid3-22.json" \
            "mid4-23 INT4+compile:cpu_int4_v3-td50-mid4-23.json" \
            "mid3-23 INT4+compile:cpu_int4_v3-td50-mid3-23.json" \
            "mid3-22 INT4+compile:cpu_int4_v3-td50-mid3-22.json" \
        --output demo.html
"""
import argparse
import base64
import json
import os
import re
import sys

# ── helpers ───────────────────────────────────────────────────────────────

def _audio_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg",
            "flac": "audio/flac", "ogg": "audio/ogg"}.get(ext, "audio/wav")
    return f"data:{mime};base64,{data}"


def _config_note(model_path: str, quant_method: str) -> str:
    """Human-readable description from model path + quant_method."""
    name = os.path.basename(model_path.rstrip("/")).replace("-tune", "")
    m = re.search(r"td(\d+)-mid(\d+)-(\d+)", name)
    if m:
        td, mid_start, n_layers = m.groups()
        prune = f"{td}% top-down pruning, mid-block from layer {mid_start}, {n_layers}-layer decoder"
    else:
        prune = name
    qmap = {
        "fp32":    "FP32, no quantization",
        "int8":    "INT8 dynamic quantization (no compile)",
        "int8ao":  "INT8 weight-only + torch.compile",
        "int4":    "INT4 weight-only + torch.compile",
    }
    q = qmap.get(quant_method.replace("_native", ""), quant_method)
    return f"{prune}; {q}"


# ── HTML template ─────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MERaLiON-2-3B CPU Benchmark Demo</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
  body { background: #f4f6fb; font-family: system-ui, sans-serif; }
  .card-section { background: #fff; border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,.07);
                  padding: 28px 32px; margin-bottom: 28px; }
  .sample-card { border: 1px solid #dee2e6; border-radius: 10px; padding: 16px 18px;
                 background: #fff; height: 100%; }
  audio { width: 100%; margin-bottom: 10px; }
  .ref-box  { background: #f8f9fa; border-left: 3px solid #adb5bd;
              padding: 7px 12px; border-radius: 4px; font-size:.9em; margin-bottom: 10px; }
  .pred-box { background: #eef2ff; border-left: 3px solid #4c6ef5;
              padding: 7px 12px; border-radius: 4px; font-size:.9em; min-height: 2.6em; }
  .lat-tag  { font-size: .78em; color: #6c757d; margin-top: 6px; }
  #paretoChart { max-height: 420px; }
  thead th { background: #212529; color: #fff; white-space: nowrap; }
  .table td, .table th { vertical-align: middle; }
  .acc-cell { font-weight: 600; }
  .best-row { background: #f0fff4 !important; }
  .footnote { color: #999; font-size: .82em; line-height: 1.7; }
  .footnote strong { color: #777; }
</style>
</head>
<body>
<div class="container py-5">

  <h1 class="fw-bold mb-1">MERaLiON-2-3B &mdash; CPU Benchmark Demo</h1>
  <p class="text-muted mb-4">Pruning &amp; quantization tradeoffs &middot; @@N_SAMPLES@@ samples &middot; IMDA PART1 ASR</p>

  <!-- ── Performance Table ──────────────────────────────────────────────── -->
  <div class="card-section">
    <h4 class="mb-3">Performance Summary</h4>
    <div class="table-responsive">
      <table class="table table-hover table-bordered align-middle mb-0">
        <thead>
          <tr>
            <th>Configuration</th>
            <th>Avg Latency</th>
            <th>Speedup vs Original</th>
            <th>Accuracy</th>
            <th>&Delta;Accuracy vs Original</th>
            <th>RAM (GB)</th>
          </tr>
        </thead>
        <tbody>@@TABLE_ROWS@@</tbody>
      </table>
    </div>
    <p class="text-muted small mt-2 mb-0">Accuracy = 1 &minus; WER, computed on normalised text (lowercase, no punctuation).</p>
  </div>

  <!-- ── Pareto Chart ───────────────────────────────────────────────────── -->
  <div class="card-section">
    <h4 class="mb-1">Pareto: Speedup vs Accuracy</h4>
    <p class="text-muted small mb-3">Upper-right is better. Each point is one configuration; the dashed line
       connects the Pareto frontier where no other option beats it on <em>both</em> axes.</p>
    <canvas id="paretoChart"></canvas>
  </div>

  <!-- ── Audio Samples ──────────────────────────────────────────────────── -->
  <div class="card-section">
    <h4 class="mb-1">Sample Transcriptions</h4>
    <p class="text-muted small mb-3">Select a configuration from the dropdown to compare transcriptions.</p>
    <div id="samplesContainer" class="row row-cols-1 row-cols-xl-2 g-3"></div>
  </div>

  <!-- ── Footnotes ──────────────────────────────────────────────────────── -->
  <div class="card-section footnote">
    <p class="mb-2"><strong>Configuration notes</strong></p>
    @@FOOTNOTES@@
  </div>

</div><!-- /container -->

<script>
/* ── data ──────────────────────────────────────────────────────────────── */
const SAMPLES       = @@SAMPLES_JSON@@;
const CONFIG_LABELS = @@CONFIG_LABELS_JSON@@;
const CHART_PTS     = @@CHART_JSON@@;
const PARETO_LINE   = @@PARETO_JSON@@;

/* ── Pareto chart ──────────────────────────────────────────────────────── */
Chart.register(ChartDataLabels);
const COLORS = ["#4361ee","#f72585","#4cc9f0","#7209b7","#06d6a0",
                "#e9c46a","#e76f51","#457b9d"];

const paretoCtx = document.getElementById("paretoChart").getContext("2d");
new Chart(paretoCtx, {{
  type: "scatter",
  data: {{
    datasets: [
      /* Pareto frontier line (no labels) */
      {{
        type: "line",
        label: "Pareto frontier",
        data: PARETO_LINE,
        borderColor: "#adb5bd",
        borderDash: [5,4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        datalabels: {{ display: false }},
      }},
      /* Individual config points */
      ...CHART_PTS.map((pt, i) => ({{
        label: pt.label,
        data: [{{ x: pt.speedup, y: pt.acc }}],
        backgroundColor: COLORS[i % COLORS.length],
        borderColor:     COLORS[i % COLORS.length],
        pointRadius: 9,
        pointHoverRadius: 12,
        datalabels: {{
          align: "top", offset: 6,
          font: {{ size: 11, weight: "600" }},
          color: COLORS[i % COLORS.length],
          formatter: () => pt.label,
        }},
      }})),
    ],
  }},
  options: {{
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: (item) => {{
            const pt = CHART_PTS[item.datasetIndex - 1];
            if (!pt) return "";
            return `${{pt.label}} — Speedup: ${{pt.speedup.toFixed(2)}}×  Accuracy: ${{pt.acc.toFixed(2)}}%`;
          }},
        }},
      }},
    }},
    scales: {{
      x: {{ title: {{ display: true, text: "Speedup (×)" }}, min: 0 }},
      y: {{ title: {{ display: true, text: "Accuracy (%)" }} }},
    }},
  }},
}});

/* ── Sample cards ──────────────────────────────────────────────────────── */
const container = document.getElementById("samplesContainer");
SAMPLES.forEach((s, idx) => {{
  const col = document.createElement("div");
  col.className = "col";

  const audioHtml = s.audio_uri
    ? `<audio controls src="${{s.audio_uri}}"></audio>`
    : `<p class="text-muted small fst-italic">Audio not available</p>`;

  const opts = CONFIG_LABELS.map(lbl =>
    `<option value="${{lbl}}">${{lbl}}</option>`
  ).join("");

  col.innerHTML = `
    <div class="sample-card">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <span class="fw-semibold text-secondary small">Sample ${{idx + 1}}</span>
        <select class="form-select form-select-sm cfg-sel" style="width:auto" data-idx="${{idx}}">
          ${{opts}}
        </select>
      </div>
      ${{audioHtml}}
      <div class="ref-box"><span class="text-muted">Reference: </span>${{s.reference}}</div>
      <div class="pred-box" id="pred-${{idx}}">—</div>
      <div class="lat-tag" id="lat-${{idx}}"></div>
    </div>`;
  container.appendChild(col);

  function showConfig(lbl) {{
    const cfg = s.configs[lbl];
    if (!cfg) return;
    document.getElementById(`pred-${{idx}}`).textContent = cfg.prediction || "(empty)";
    document.getElementById(`lat-${{idx}}`).textContent  = `⏱ ${{cfg.latency_s.toFixed(1)}}s`;
  }}
  col.querySelector(".cfg-sel").addEventListener("change", e => showConfig(e.target.value));
  showConfig(CONFIG_LABELS[0]);
}});
</script>
</body>
</html>
"""


# ── build ─────────────────────────────────────────────────────────────────

def _pareto_frontier(points):
    """Pareto frontier for (speedup, acc) — both higher is better.

    Sort by speedup ascending; a point is non-dominated if no other point
    has both higher speedup AND higher accuracy.
    """
    non_dom = []
    for p in points:
        dominated = any(
            q["speedup"] >= p["speedup"] and q["acc"] >= p["acc"]
            and (q["speedup"] > p["speedup"] or q["acc"] > p["acc"])
            for q in points if q is not p
        )
        if not dominated:
            non_dom.append(p)
    return sorted(non_dom, key=lambda p: p["speedup"])


def build_html(configs: dict, n_samples: int) -> str:
    # ── rename configs: first → "Original Model", rest → A, B, C, … ──────
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    orig_labels = list(configs.keys())
    label_map = {}
    for i, lbl in enumerate(orig_labels):
        label_map[lbl] = "Original Model" if i == 0 else letters[i - 1]

    base_data = next(iter(configs.values()))
    base_lat  = base_data.get("avg_latency_s", 1.0)
    base_acc  = (1 - base_data.get("wer", 0.0)) * 100

    # ── summary table rows ────────────────────────────────────────────────
    table_rows = ""
    for i, (orig_lbl, data) in enumerate(configs.items()):
        disp  = label_map[orig_lbl]
        lat   = data.get("avg_latency_s", 0)
        acc   = (1 - data.get("wer", float("nan"))) * 100
        ram   = data.get("ram_mb", 0) / 1024
        spd   = base_lat / lat if lat > 0 else 0
        dacc  = acc - base_acc
        row_cls = ' class="best-row"' if i == 0 else ""
        if i == 0:
            dacc_str = "—"
            spd_str  = "—"
        else:
            # accuracy drop → grey (not red); accuracy gain → green
            color = "success" if dacc > 0 else "secondary"
            dacc_str = f'<span class="text-{color}">{dacc:+.2f}%</span>'
            spd_str  = f"{spd:.2f}×"
        table_rows += f"""
        <tr{row_cls}>
          <td><strong>{disp}</strong></td>
          <td>{lat:.2f} s</td>
          <td>{spd_str}</td>
          <td class="acc-cell">{acc:.2f}%</td>
          <td>{dacc_str}</td>
          <td>{ram:.2f} GB</td>
        </tr>"""

    # ── chart data ────────────────────────────────────────────────────────
    chart_pts = [
        {"label":   label_map[lbl],
         "speedup": base_lat / data.get("avg_latency_s", 1),
         "acc":     (1 - data.get("wer", 0)) * 100}
        for lbl, data in configs.items()
    ]
    pareto     = _pareto_frontier(chart_pts)
    pareto_line = [{"x": p["speedup"], "y": p["acc"]} for p in pareto]

    # ── samples data (embedded audio) ─────────────────────────────────────
    ref_data = next(
        (d for d in configs.values() if d.get("samples")), None
    )
    if ref_data is None:
        print("WARNING: No config has 'samples' data. Audio cards will be empty.",
              file=sys.stderr)
        ref_data = {"samples": [{"idx": i, "reference": "", "audio_file": None}
                                for i in range(n_samples)]}

    samples_js = []
    for i, ref_s in enumerate(ref_data["samples"]):
        audio_uri = ""
        afile = ref_s.get("audio_file")
        if afile and os.path.exists(afile):
            try:
                audio_uri = _audio_data_uri(afile)
            except Exception as e:
                print(f"  WARNING: could not embed {afile}: {e}", file=sys.stderr)

        cfg_data = {}
        for orig_lbl, data in configs.items():
            disp  = label_map[orig_lbl]
            slist = data.get("samples", [])
            if i < len(slist):
                s = slist[i]
                cfg_data[disp] = {
                    "prediction": s.get("prediction", ""),
                    "latency_s":  s.get("latency_s", 0.0),
                }
        samples_js.append({
            "audio_uri": audio_uri,
            "reference": ref_s.get("reference", ""),
            "configs":   cfg_data,
        })

    # ── footnotes ─────────────────────────────────────────────────────────
    footnote_items = []
    for i, (orig_lbl, data) in enumerate(configs.items()):
        disp = label_map[orig_lbl]
        if disp == "Original Model":
            continue
        note = _config_note(
            data.get("model", orig_lbl),
            data.get("quant_method", "")
        )
        footnote_items.append(f"<li><strong>{disp}</strong> &mdash; {note}</li>")
    footnotes_html = "<ul style='list-style:none;padding:0;margin:0'>" \
                     + "".join(footnote_items) + "</ul>"

    # ── render ────────────────────────────────────────────────────────────
    # Step 1: unescape {{ / }} in template BEFORE inserting JSON data.
    html = _HTML.replace("{{", "{").replace("}}", "}")
    # Step 2: substitute tokens — inserted JSON uses plain { } and is safe.
    html = (html
        .replace("@@N_SAMPLES@@",          str(n_samples))
        .replace("@@TABLE_ROWS@@",         table_rows)
        .replace("@@SAMPLES_JSON@@",       json.dumps(samples_js, ensure_ascii=False))
        .replace("@@CONFIG_LABELS_JSON@@", json.dumps(list(label_map.values())))
        .replace("@@CHART_JSON@@",         json.dumps(chart_pts, ensure_ascii=False))
        .replace("@@PARETO_JSON@@",        json.dumps(pareto_line))
        .replace("@@FOOTNOTES@@",          footnotes_html)
    )
    return html


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML demo page from CPU benchmark JSON results")
    parser.add_argument("--configs", nargs="+", required=True,
                        help='"Label:json_file" pairs in display order')
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override sample count shown in subtitle")
    parser.add_argument("--output", default="demo.html")
    args = parser.parse_args()

    configs = {}
    for spec in args.configs:
        colon = spec.index(":")
        label, json_file = spec[:colon], spec[colon + 1:]
        if not os.path.exists(json_file):
            print(f"WARNING: {json_file} not found — skipping '{label}'", file=sys.stderr)
            continue
        with open(json_file) as f:
            configs[label] = json.load(f)

    if not configs:
        sys.exit("No valid config files found.")

    n = args.num_samples or next(iter(configs.values())).get("num_samples", "?")
    html = build_html(configs, n)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Demo saved → {args.output}  ({os.path.getsize(args.output) // 1024} KB)")


if __name__ == "__main__":
    main()
