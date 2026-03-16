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
import sys

# ── helpers ───────────────────────────────────────────────────────────────

def _audio_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg",
            "flac": "audio/flac", "ogg": "audio/ogg"}.get(ext, "audio/wav")
    return f"data:{mime};base64,{data}"


# ── HTML template ─────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MERaLiON-2 CPU Benchmark Demo</title>
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
  #paretoChart { max-height: 400px; }
  thead th { background: #212529; color: #fff; white-space: nowrap; }
  .table td, .table th { vertical-align: middle; }
  .wer-cell { font-weight: 600; }
  .best-row { background: #f0fff4 !important; }
</style>
</head>
<body>
<div class="container py-5">

  <h1 class="fw-bold mb-1">MERaLiON-2 — CPU Benchmark Demo</h1>
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
            <th>Speedup vs FP32</th>
            <th>WER</th>
            <th>ΔWER vs FP32</th>
            <th>RAM</th>
          </tr>
        </thead>
        <tbody>@@TABLE_ROWS@@</tbody>
      </table>
    </div>
    <p class="text-muted small mt-2 mb-0">WER computed on normalised text (lowercase, no punctuation).</p>
  </div>

  <!-- ── Pareto Chart ───────────────────────────────────────────────────── -->
  <div class="card-section">
    <h4 class="mb-1">Pareto: Latency vs WER</h4>
    <p class="text-muted small mb-3">Lower-left is better. Each point is one configuration; the Pareto frontier
       marks configurations where no other option beats it on <em>both</em> axes.</p>
    <canvas id="paretoChart"></canvas>
  </div>

  <!-- ── Audio Samples ──────────────────────────────────────────────────── -->
  <div class="card-section">
    <h4 class="mb-1">Sample Transcriptions</h4>
    <p class="text-muted small mb-3">Select a configuration from the dropdown to compare transcriptions.</p>
    <div id="samplesContainer" class="row row-cols-1 row-cols-xl-2 g-3"></div>
  </div>

</div><!-- /container -->

<script>
/* ── data ──────────────────────────────────────────────────────────────── */
const SAMPLES      = @@SAMPLES_JSON@@;
const CONFIG_LABELS = @@CONFIG_LABELS_JSON@@;
const CHART_PTS    = @@CHART_JSON@@;
const PARETO_LINE  = @@PARETO_JSON@@;

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
        data: [{{ x: pt.wer, y: pt.lat }}],
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
            return `${{pt.label}} — WER: ${{pt.wer.toFixed(2)}}%  lat: ${{pt.lat.toFixed(2)}}s`;
          }},
        }},
      }},
    }},
    scales: {{
      x: {{ title: {{ display: true, text: "WER (%)" }}, min: 0 }},
      y: {{ title: {{ display: true, text: "Avg Latency (s)" }}, min: 0 }},
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
    """Return points on the Pareto frontier (min WER, min latency), sorted by WER."""
    pts = sorted(points, key=lambda p: p["wer"])
    frontier = []
    best_lat = float("inf")
    for p in pts:
        if p["lat"] < best_lat:
            frontier.append(p)
            best_lat = p["lat"]
    return frontier


def build_html(configs: dict, n_samples: int) -> str:
    base_lat = next(iter(configs.values())).get("avg_latency_s", 1.0)
    base_wer = next(iter(configs.values())).get("wer", 0.0) * 100

    # ── summary table rows ────────────────────────────────────────────────
    table_rows = ""
    for i, (label, data) in enumerate(configs.items()):
        lat  = data.get("avg_latency_s", 0)
        wer  = data.get("wer", float("nan")) * 100
        ram  = data.get("ram_mb", 0)
        spd  = base_lat / lat if lat > 0 else 0
        dwer = wer - base_wer
        best = (i == 0)
        row_cls = ' class="best-row"' if best else ""
        dwer_str = ("—" if i == 0
                    else f'<span class="text-{"danger" if dwer>0 else "success"}">'
                         f'{dwer:+.2f}%</span>')
        table_rows += f"""
        <tr{row_cls}>
          <td><strong>{label}</strong></td>
          <td>{lat:.2f} s</td>
          <td>{"—" if i == 0 else f"{spd:.2f}×"}</td>
          <td class="wer-cell">{wer:.2f}%</td>
          <td>{dwer_str}</td>
          <td>{ram:.0f} MB</td>
        </tr>"""

    # ── chart data ────────────────────────────────────────────────────────
    chart_pts = [
        {"label": label,
         "wer": data.get("wer", 0) * 100,
         "lat": data.get("avg_latency_s", 0)}
        for label, data in configs.items()
    ]
    pareto = _pareto_frontier(chart_pts)
    # extend pareto line to y-axis for visual clarity
    pareto_line = [{"x": p["wer"], "y": p["lat"]} for p in pareto]

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
        for label, data in configs.items():
            slist = data.get("samples", [])
            if i < len(slist):
                s = slist[i]
                cfg_data[label] = {
                    "prediction": s.get("prediction", ""),
                    "latency_s":  s.get("latency_s", 0.0),
                }
        samples_js.append({
            "audio_uri": audio_uri,
            "reference": ref_s.get("reference", ""),
            "configs":   cfg_data,
        })

    return (
        _HTML
        .replace("@@N_SAMPLES@@",        str(n_samples))
        .replace("@@TABLE_ROWS@@",       table_rows)
        .replace("@@SAMPLES_JSON@@",     json.dumps(samples_js, ensure_ascii=False))
        .replace("@@CONFIG_LABELS_JSON@@", json.dumps(list(configs.keys())))
        .replace("@@CHART_JSON@@",       json.dumps(chart_pts, ensure_ascii=False))
        .replace("@@PARETO_JSON@@",      json.dumps(pareto_line))
    )


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
