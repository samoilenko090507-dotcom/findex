import pstats
from pathlib import Path

ps = pstats.Stats("search_profile.prof")
stats_dict = getattr(ps, "stats", {})
top_calls = sorted(stats_dict.items(), key=lambda item: item[1][3], reverse=True)[:28]

bars = []
for i, (k, v) in enumerate(top_calls):
    fn = str(k[2]).replace("<", "&lt;").replace(">", "&gt;")
    file_short = Path(str(k[0])).name
    time_val = float(v[3])
    calls = int(v[0])
    bar_w = max(30, min(1120, int(time_val * 35000) + 90))
    y = 70 + i * 18
    r = min(255, 190 + i * 2)
    g = max(40, 95 - i * 2)
    b = max(40, 60 + i * 4)
    bars.append(
        f'<rect class="bar" x="20" y="{y}" width="{bar_w}" height="15" fill="rgb({r},{g},{b})"/>'
        f'<text x="30" y="{y+12}">{fn} ({file_short}:{k[1]}) – {time_val:.4f}s [{calls} calls]</text>'
    )

svg_content = f"""<svg version="1.1" width="1200" height="600" xmlns="http://www.w3.org/2000/svg">
<style>
text {{ font-family: Consolas, monospace; font-size: 11px; fill: #ffffff; }}
.bar {{ rx: 3; ry: 3; stroke: #181825; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="#11111b"/>
<text x="20" y="35" font-size="16" font-weight="bold" fill="#89b4fa">findex cProfile Flame Graph – Search 3-terms BM25</text>
{''.join(bars)}
</svg>"""

out_dir = Path("docs")
out_dir.mkdir(parents=True, exist_ok=True)
Path("docs/flamegraph.svg").write_text(svg_content, encoding="utf-8")
print("Flame graph successfully generated: docs/flamegraph.svg")
