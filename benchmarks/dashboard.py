"""Benchmark dashboard — generates interactive HTML reports from benchmark results.

Produces a self-contained HTML dashboard with:
- Throughput comparison charts
- Latency distribution plots
- Memory scaling curves
- Delayed recall heatmaps
- Summary statistics tables

Usage:
    python -m benchmarks.dashboard --results-dir ./benchmark_results
    python -m benchmarks.dashboard --results-dir ./benchmark_results --output dashboard.html
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptive KV Memory — Benchmark Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #1c2128;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
            --border: #30363d;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }}
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header .subtitle {{ color: var(--text-secondary); font-size: 1.1rem; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 3rem;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-green);
        }}
        .stat-card .label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }}

        .section {{
            margin-bottom: 3rem;
        }}
        .section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--accent-blue);
        }}

        .chart-container {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}
        @media (max-width: 900px) {{
            .chart-row {{ grid-template-columns: 1fr; }}
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--bg-secondary);
            color: var(--accent-blue);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        td {{ font-size: 0.9rem; }}
        tr:last-child td {{ border-bottom: none; }}
        .highlight {{ color: var(--accent-green); font-weight: 600; }}
        .warn {{ color: var(--accent-yellow); }}
        .bad {{ color: var(--accent-red); }}

        .heatmap {{
            display: grid;
            gap: 2px;
            margin-top: 1rem;
        }}
        .heatmap-cell {{
            padding: 0.5rem;
            text-align: center;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            color: var(--text-secondary);
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Adaptive KV Memory</h1>
        <div class="subtitle">Benchmark Dashboard — Three-Tier Hierarchical KV Cache</div>
    </div>

    <div class="summary-grid">
        <div class="stat-card">
            <div class="value">{vram_savings_pct}%</div>
            <div class="label">VRAM Savings (avg)</div>
        </div>
        <div class="stat-card">
            <div class="value">{throughput_ratio}x</div>
            <div class="label">Throughput vs Baseline</div>
        </div>
        <div class="stat-card">
            <div class="value">{ppl_ratio}</div>
            <div class="label">PPL Ratio (lower=better)</div>
        </div>
        <div class="stat-card">
            <div class="value">{recall_accuracy}%</div>
            <div class="label">Delayed Recall Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="value">{max_context}K</div>
            <div class="label">Max Context (no OOM)</div>
        </div>
    </div>

    <div class="section">
        <h2>Throughput Comparison</h2>
        <div class="chart-container">
            <canvas id="throughputChart" height="80"></canvas>
        </div>
    </div>

    <div class="section">
        <h2>Latency Profile</h2>
        <div class="chart-row">
            <div class="chart-container">
                <canvas id="ttftChart" height="120"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="itlChart" height="120"></canvas>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Memory Scaling</h2>
        <div class="chart-container">
            <canvas id="memoryChart" height="80"></canvas>
        </div>
    </div>

    <div class="section">
        <h2>Delayed Recall — Passkey Retrieval</h2>
        <div class="chart-container">
            <canvas id="recallChart" height="80"></canvas>
        </div>
    </div>

    <div class="section">
        <h2>Detailed Results</h2>
        {results_table}
    </div>

    <div class="footer">
        Generated by Adaptive KV Memory benchmark suite<br>
        {timestamp}
    </div>

    <script>
        const COLORS = {{
            'Full Cache': '#8b949e',
            'AKV-4bit': '#58a6ff',
            'AKV-2bit': '#a371f7',
            'H2O': '#f85149',
            'KIVI': '#d29922',
            'SnapKV': '#3fb950',
            'ScissorHands': '#f778ba',
        }};

        function getColor(method) {{
            for (const [key, color] of Object.entries(COLORS)) {{
                if (method.includes(key)) return color;
            }}
            return '#8b949e';
        }}

        // Throughput Chart
        const throughputData = {throughput_data};
        if (throughputData.length > 0) {{
            const methods = [...new Set(throughputData.map(d => d.method))];
            const seqLens = [...new Set(throughputData.map(d => d.seq_len))].sort((a,b) => a-b);

            new Chart(document.getElementById('throughputChart'), {{
                type: 'bar',
                data: {{
                    labels: seqLens.map(s => s + ' tokens'),
                    datasets: methods.map(m => ({{
                        label: m,
                        data: seqLens.map(s => {{
                            const d = throughputData.find(x => x.method === m && x.seq_len === s);
                            return d ? d.decode_tokens_per_sec : 0;
                        }}),
                        backgroundColor: getColor(m) + '99',
                        borderColor: getColor(m),
                        borderWidth: 1,
                    }})),
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Decode Throughput (tok/s)', color: '#e6edf3' }} }},
                    scales: {{
                        y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                        x: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    }},
                }},
            }});
        }}

        // Memory Chart
        const memoryData = {memory_data};
        if (memoryData.length > 0) {{
            const methods = [...new Set(memoryData.map(d => d.method))];
            const seqLens = [...new Set(memoryData.map(d => d.seq_len))].sort((a,b) => a-b);

            new Chart(document.getElementById('memoryChart'), {{
                type: 'line',
                data: {{
                    labels: seqLens.map(s => s/1024 + 'K'),
                    datasets: methods.map(m => ({{
                        label: m,
                        data: seqLens.map(s => {{
                            const d = memoryData.find(x => x.method === m && x.seq_len === s);
                            return d ? d.memory_mb : null;
                        }}),
                        borderColor: getColor(m),
                        backgroundColor: getColor(m) + '22',
                        fill: false,
                        tension: 0.3,
                    }})),
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'VRAM Usage vs Context Length (MB)', color: '#e6edf3' }} }},
                    scales: {{
                        y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                        x: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    }},
                }},
            }});
        }}

        // TTFT Chart
        const latencyData = {latency_data};
        if (latencyData.length > 0) {{
            const methods = [...new Set(latencyData.map(d => d.method))];
            const seqLens = [...new Set(latencyData.map(d => d.seq_len))].sort((a,b) => a-b);

            new Chart(document.getElementById('ttftChart'), {{
                type: 'bar',
                data: {{
                    labels: seqLens.map(s => s + ' tokens'),
                    datasets: methods.map(m => ({{
                        label: m,
                        data: seqLens.map(s => {{
                            const d = latencyData.find(x => x.method === m && x.seq_len === s);
                            return d ? d.ttft_ms : 0;
                        }}),
                        backgroundColor: getColor(m) + '99',
                        borderColor: getColor(m),
                        borderWidth: 1,
                    }})),
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Time to First Token (ms)', color: '#e6edf3' }} }},
                    scales: {{
                        y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                        x: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    }},
                }},
            }});

            // ITL Chart
            new Chart(document.getElementById('itlChart'), {{
                type: 'bar',
                data: {{
                    labels: seqLens.map(s => s + ' tokens'),
                    datasets: methods.map(m => ({{
                        label: m + ' (p50)',
                        data: seqLens.map(s => {{
                            const d = latencyData.find(x => x.method === m && x.seq_len === s);
                            return d ? d.itl_p50_ms : 0;
                        }}),
                        backgroundColor: getColor(m) + '99',
                        borderColor: getColor(m),
                        borderWidth: 1,
                    }})),
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Inter-Token Latency p50 (ms)', color: '#e6edf3' }} }},
                    scales: {{
                        y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                        x: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    }},
                }},
            }});
        }}

        // Recall Chart
        const recallData = {recall_data};
        if (recallData.length > 0) {{
            const methods = [...new Set(recallData.map(d => d.method))];
            const positions = [...new Set(recallData.map(d => d.needle_position))].sort((a,b) => a-b);

            new Chart(document.getElementById('recallChart'), {{
                type: 'line',
                data: {{
                    labels: positions.map(p => (p*100).toFixed(0) + '% depth'),
                    datasets: methods.map(m => ({{
                        label: m,
                        data: positions.map(p => {{
                            const matches = recallData.filter(x => x.method === m && x.needle_position === p);
                            return matches.length ? matches.reduce((s,x) => s + x.accuracy, 0) / matches.length : null;
                        }}),
                        borderColor: getColor(m),
                        backgroundColor: getColor(m) + '22',
                        fill: false,
                        tension: 0.3,
                    }})),
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Passkey Retrieval Accuracy by Depth', color: '#e6edf3' }} }},
                    scales: {{
                        y: {{ min: 0, max: 1, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e', callback: v => (v*100)+'%' }} }},
                        x: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    }},
                }},
            }});
        }}
    </script>
</body>
</html>"""


class BenchmarkDashboard:
    """Generates an interactive HTML dashboard from benchmark results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.throughput_data = []
        self.latency_data = []
        self.memory_data = []
        self.recall_data = []

    def load_results(self):
        """Load all JSON result files from the results directory."""
        if not self.results_dir.exists():
            logger.warning(f"Results directory not found: {self.results_dir}")
            return

        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if "throughput" in json_file.name:
                    self.throughput_data.extend(data.get("results", []))
                elif "latency" in json_file.name:
                    self.latency_data.extend(data.get("results", []))
                elif "memory" in json_file.name:
                    self.memory_data.extend(data.get("results", []))
                elif "recall" in json_file.name or "delayed" in json_file.name:
                    self.recall_data.extend(data.get("results", []))

                logger.info(f"Loaded: {json_file.name}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    def _compute_summary_stats(self) -> dict:
        """Compute headline statistics."""
        stats = {
            "vram_savings_pct": "—",
            "throughput_ratio": "—",
            "ppl_ratio": "—",
            "recall_accuracy": "—",
            "max_context": "—",
        }

        if self.memory_data:
            akv_mem = [d.get("memory_mb", 0) for d in self.memory_data if "AKV" in d.get("method", "")]
            full_mem = [d.get("memory_mb", 0) for d in self.memory_data if "Full" in d.get("method", "")]
            if akv_mem and full_mem:
                avg_akv = sum(akv_mem) / len(akv_mem)
                avg_full = sum(full_mem) / len(full_mem)
                if avg_full > 0:
                    stats["vram_savings_pct"] = f"{(1 - avg_akv/avg_full)*100:.0f}"

        if self.throughput_data:
            akv_tp = [d.get("decode_tokens_per_sec", 0) for d in self.throughput_data if "AKV" in d.get("method", "")]
            full_tp = [d.get("decode_tokens_per_sec", 0) for d in self.throughput_data if "Full" in d.get("method", "")]
            if akv_tp and full_tp:
                avg_akv = sum(akv_tp) / len(akv_tp)
                avg_full = sum(full_tp) / len(full_tp)
                if avg_full > 0:
                    stats["throughput_ratio"] = f"{avg_akv/avg_full:.2f}"

        if self.recall_data:
            akv_acc = [d.get("accuracy", 0) for d in self.recall_data if "akv" in d.get("method", "").lower()]
            if akv_acc:
                stats["recall_accuracy"] = f"{sum(akv_acc)/len(akv_acc)*100:.0f}"

        return stats

    def _generate_results_table(self) -> str:
        """Generate HTML table from throughput results."""
        if not self.throughput_data:
            return "<p>No throughput data available. Run benchmarks first.</p>"

        rows = ""
        for d in self.throughput_data[:20]:  # Limit to first 20
            method = d.get("method", "Unknown")
            decode = d.get("decode_tokens_per_sec", 0)
            vram = d.get("vram_peak_mb", 0)
            seq_len = d.get("seq_len", 0)
            css_class = "highlight" if "AKV" in method else ""
            rows += f"""<tr>
                <td class="{css_class}">{method}</td>
                <td>{seq_len:,}</td>
                <td>{decode:.1f}</td>
                <td>{vram:.0f}</td>
            </tr>"""

        return f"""<table>
            <thead><tr><th>Method</th><th>Seq Len</th><th>Decode tok/s</th><th>VRAM MB</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    def generate(self, output_path: Optional[str] = None) -> str:
        """Generate the dashboard HTML."""
        import datetime

        self.load_results()
        stats = self._compute_summary_stats()

        html = DASHBOARD_HTML_TEMPLATE.format(
            vram_savings_pct=stats["vram_savings_pct"],
            throughput_ratio=stats["throughput_ratio"],
            ppl_ratio=stats.get("ppl_ratio", "≤1.02"),
            recall_accuracy=stats["recall_accuracy"],
            max_context=stats.get("max_context", "32"),
            throughput_data=json.dumps(self.throughput_data),
            memory_data=json.dumps(self.memory_data),
            latency_data=json.dumps(self.latency_data),
            recall_data=json.dumps(self.recall_data),
            results_table=self._generate_results_table(),
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        if output_path is None:
            output_path = str(self.results_dir / "dashboard.html")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Dashboard generated: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark dashboard")
    parser.add_argument("--results-dir", default="./benchmark_results",
                        help="Directory containing benchmark JSON results")
    parser.add_argument("--output", default=None,
                        help="Output HTML file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    dashboard = BenchmarkDashboard(args.results_dir)
    path = dashboard.generate(args.output)
    print(f"Dashboard: {path}")


if __name__ == "__main__":
    main()
