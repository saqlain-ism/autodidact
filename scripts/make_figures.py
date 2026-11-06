"""Draw the README and site charts from the committed results.

    python scripts/make_figures.py

Reads results/docs-only and results/learned and writes plain SVG files to
docs/figures/. No plotting library is needed, so the charts can always be
regenerated from the raw data.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

PACK = {"restock_one", "restock_category", "fulfil_or_backorder"}
BEFORE, AFTER = "#9aa1ad", "#1f6feb"
INK, MUTED, GRID, PAPER = "#1f2328", "#59636e", "#e6e8eb", "#ffffff"
FONT = "font-family=\"-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\""


def load(name: str) -> list[dict]:
    path = RESULTS / name / "results.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def rate(rows: list[dict]) -> tuple[int, int]:
    return sum(r["passed"] for r in rows), len(rows)


def text(x, y, s, size=13, color=INK, anchor="start", weight="400") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}" {FONT}>{escape(str(s))}</text>'
    )


def svg(width: int, height: int, body: list[str], title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>"
        f'<rect width="{width}" height="{height}" rx="8" fill="{PAPER}"/>'
        + "".join(body)
        + "</svg>"
    )


def legend(x: int, y: int) -> list[str]:
    return [
        f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="2" fill="{BEFORE}"/>',
        text(x + 18, y, "No practice (tool docs only)", 12, MUTED),
        f'<rect x="{x + 210}" y="{y - 10}" width="12" height="12" rx="2" fill="{AFTER}"/>',
        text(x + 228, y, "After self-directed practice", 12, MUTED),
    ]


def headline(before: list[dict], after: list[dict]) -> str:
    groups = [
        (
            "Tasks affected by the rule it discovered",
            [r for r in before if r["template"] in PACK],
            [r for r in after if r["template"] in PACK],
        ),
        ("All hidden tasks", before, after),
    ]
    w, left, right = 720, 24, 64
    plot = w - left - right
    body = [
        text(left, 34, "Success on unseen tasks", 18, INK, weight="600"),
        text(
            left,
            56,
            "Same model, same hidden tasks; only the agent's self-written notebook differs.",
            12,
            MUTED,
        ),
    ]
    body += legend(left, 84)
    y = 120
    for label, b, a in groups:
        body.append(text(left, y, label, 13, INK, weight="600"))
        y += 10
        for rows, color in ((b, BEFORE), (a, AFTER)):
            p, n = rate(rows)
            bw = plot * p / n
            body.append(
                f'<rect x="{left}" y="{y}" width="{plot}" height="22" rx="4" fill="{GRID}"/>'
            )
            body.append(
                f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="22" rx="4" fill="{color}"/>'
            )
            body.append(text(left + plot + 8, y + 16, f"{100 * p / n:.1f}%", 13, INK, weight="600"))
            body.append(text(left + 8, y + 16, f"{p}/{n}", 12, PAPER if bw > 60 else INK))
            y += 28
        y += 22
    return svg(w, y, body, "Success on unseen tasks before and after practice")


def per_template(before: list[dict], after: list[dict]) -> str:
    names = sorted({r["template"] for r in before})
    stats = []
    for t in names:
        pb, nb = rate([r for r in before if r["template"] == t])
        pa, na = rate([r for r in after if r["template"] == t])
        stats.append((t, pb / nb, pa / na, f"{pb}/{nb} → {pa}/{na}"))
    stats.sort(key=lambda s: (-(s[2] - s[1]), s[0]))
    w, label_w, right = 720, 190, 110
    x0, plot = label_w, w - label_w - right
    row = 26
    top = 92
    h = top + row * len(stats) + 20
    body = [
        text(24, 34, "Pass rate by task type", 18, INK, weight="600"),
        text(
            24, 56, "Bold rows depend on the case-pack rule the agent found in practice.", 12, MUTED
        ),
    ]
    body += legend(24, 80)
    for frac in (0, 0.5, 1):
        gx = x0 + plot * frac
        body.append(f'<line x1="{gx}" y1="{top - 6}" x2="{gx}" y2="{h - 18}" stroke="{GRID}"/>')
        body.append(text(gx, h - 4, f"{int(frac * 100)}%", 11, MUTED, "middle"))
    for i, (t, b, a, label) in enumerate(stats):
        y = top + i * row + 10
        weight = "700" if t in PACK else "400"
        body.append(text(x0 - 12, y + 4, t.replace("_", " "), 12, INK, "end", weight))
        xb, xa = x0 + plot * b, x0 + plot * a
        if abs(xa - xb) > 0.5:
            color = AFTER if a > b else BEFORE
            body.append(
                f'<line x1="{xb}" y1="{y}" x2="{xa}" y2="{y}" stroke="{color}" stroke-width="3"/>'
            )
        body.append(f'<circle cx="{xb}" cy="{y}" r="6" fill="{BEFORE}"/>')
        body.append(
            f'<circle cx="{xa}" cy="{y}" r="6" fill="{AFTER}" stroke="{PAPER}" stroke-width="1.5"/>'
        )
        body.append(text(x0 + plot + 14, y + 4, label, 11, MUTED))
    return svg(w, h, body, "Pass rate by task type before and after practice")


def main() -> None:
    before, after = load("docs-only"), load("learned")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "headline.svg").write_text(headline(before, after), encoding="utf-8", newline="\n")
    (OUT / "by-task-type.svg").write_text(
        per_template(before, after), encoding="utf-8", newline="\n"
    )
    print(f"wrote {OUT / 'headline.svg'} and {OUT / 'by-task-type.svg'}")


if __name__ == "__main__":
    main()
