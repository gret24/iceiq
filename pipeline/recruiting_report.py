"""
IceIQ Recruiting Report Generator
Generates per-player PDF scouting reports with stats, charts, and scout notes.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ── Brand colours ────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1A3A5C")
ICE_BLUE   = colors.HexColor("#00A8E8")
LIGHT_GREY = colors.HexColor("#F4F6F9")
MID_GREY   = colors.HexColor("#D0D5DD")
WHITE      = colors.white
BLACK      = colors.black

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class PlayerReport:
    name:              str
    team:              str
    position:          str
    shot_hand:         str

    avg_speed_kmh:     float
    max_speed_kmh:     float
    total_distance_m:  float
    sprint_count:      int

    frame_count:       int                        # frames @ 30 fps → TOI
    zone_time:         Dict[str, float]           # {def, neu, off} seconds
    events:            Dict[str, int]             # zone_entries, faceoffs, sprints, goal_area

    @property
    def toi_seconds(self) -> float:
        return self.frame_count / 30.0

    @property
    def toi_str(self) -> str:
        t = int(self.toi_seconds)
        return f"{t // 60}:{t % 60:02d}"


# ── Matplotlib helper charts ──────────────────────────────────────────────────

def _fig_to_image(fig, width_in: float, height_in: float) -> Image:
    """Render a matplotlib figure to a ReportLab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_in * inch, height=height_in * inch)


def _speed_bar_chart(player: PlayerReport, width_in=2.2, height_in=2.0) -> Image:
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor("#F4F6F9")
    ax.set_facecolor("#F4F6F9")

    labels = ["Avg Speed\n(km/h)", "Max Speed\n(km/h)"]
    values = [player.avg_speed_kmh, player.max_speed_kmh]
    bar_colors = ["#00A8E8", "#1A3A5C"]
    bars = ax.bar(labels, values, color=bar_colors, width=0.5, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#1A3A5C")

    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Speed", fontsize=10, fontweight="bold", color="#1A3A5C", pad=6)
    ax.yaxis.set_visible(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", colors="#1A3A5C", labelsize=8)
    ax.grid(axis="y", color="#D0D5DD", linewidth=0.5, zorder=0)
    fig.tight_layout(pad=0.4)
    return _fig_to_image(fig, width_in, height_in)


def _zone_donut(player: PlayerReport, width_in=2.2, height_in=2.0) -> Image:
    zone = player.zone_time
    total = sum(zone.values()) or 1
    labels = ["DEF", "NEU", "OFF"]
    sizes  = [zone.get("def", 0), zone.get("neu", 0), zone.get("off", 0)]
    pcts   = [s / total * 100 for s in sizes]
    chart_colors = ["#1A3A5C", "#00A8E8", "#90CAF9"]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor("#F4F6F9")
    ax.set_facecolor("#F4F6F9")

    wedges, _ = ax.pie(
        sizes, colors=chart_colors, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=1.5),
    )
    ax.set_title("Zone Time", fontsize=10, fontweight="bold", color="#1A3A5C", pad=6)

    legend_labels = [f"{l} {p:.0f}%" for l, p in zip(labels, pcts)]
    ax.legend(wedges, legend_labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=7, frameon=False)
    fig.tight_layout(pad=0.4)
    return _fig_to_image(fig, width_in, height_in)


def _ice_heatmap(player: PlayerReport, width_in=2.8, height_in=2.0) -> Image:
    """
    Schematic ice rink heatmap.  We use zone_time proportions to weight
    synthetic density across the three zones (defensive-left → offensive-right).
    """
    zone = player.zone_time
    total = sum(zone.values()) or 1
    def_w = zone.get("def", 0) / total
    neu_w = zone.get("neu", 0) / total
    off_w = zone.get("off", 0) / total

    nx, ny = 90, 40
    data = np.zeros((ny, nx))
    # Each column x maps to a zone
    for x in range(nx):
        frac = x / nx
        if frac < 1 / 3:
            weight = def_w
        elif frac < 2 / 3:
            weight = neu_w
        else:
            weight = off_w
        noise = np.random.RandomState(x + int(player.avg_speed_kmh * 10)).rand(ny)
        data[:, x] = weight * (0.5 + 0.5 * noise)

    # Smooth with a simple gaussian
    from scipy.ndimage import gaussian_filter  # optional; fall back gracefully
    try:
        data = gaussian_filter(data, sigma=4)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor("#F4F6F9")

    ax.imshow(data, cmap="Blues", aspect="auto", origin="lower",
              extent=[0, nx, 0, ny], vmin=0)

    # Draw simplified rink outline
    rink = mpatches.FancyBboxPatch((1, 1), nx - 2, ny - 2,
                                   boxstyle="round,pad=2",
                                   linewidth=1.5, edgecolor="#1A3A5C",
                                   facecolor="none", zorder=5)
    ax.add_patch(rink)
    # Centre line
    ax.axvline(nx / 2, color="#1A3A5C", linewidth=1, linestyle="--", zorder=5)
    # Blue lines
    ax.axvline(nx / 3, color="#00A8E8", linewidth=1, zorder=5)
    ax.axvline(2 * nx / 3, color="#00A8E8", linewidth=1, zorder=5)
    # Goal crease circles
    for xc in [6, nx - 6]:
        circle = plt.Circle((xc, ny / 2), 3, color="#1A3A5C",
                             fill=False, linewidth=1, zorder=5)
        ax.add_patch(circle)

    ax.set_title("Ice Heat Map", fontsize=10, fontweight="bold", color="#1A3A5C", pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return _fig_to_image(fig, width_in, height_in)


# ── ReportLab page builder ────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "IQTitle", parent=base["Title"],
            textColor=WHITE, fontSize=18, leading=22,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "IQSubtitle", parent=base["Normal"],
            textColor=ICE_BLUE, fontSize=10, leading=13,
            fontName="Helvetica", alignment=TA_LEFT,
        ),
        "section": ParagraphStyle(
            "IQSection", parent=base["Normal"],
            textColor=DARK_BLUE, fontSize=10, leading=14,
            fontName="Helvetica-Bold", spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "IQBody", parent=base["Normal"],
            textColor=BLACK, fontSize=9, leading=13,
            fontName="Helvetica",
        ),
        "note_line": ParagraphStyle(
            "IQNote", parent=base["Normal"],
            textColor=MID_GREY, fontSize=9, leading=18,
            fontName="Helvetica",
        ),
    }
    return styles


def _header_footer(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, doc.pagesize[1] - 0.55 * inch,
                doc.pagesize[0], 0.55 * inch, fill=1, stroke=0)
    # IceIQ wordmark
    canvas.setFillColor(ICE_BLUE)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(0.4 * inch, doc.pagesize[1] - 0.38 * inch, "IceIQ")
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(doc.pagesize[0] - 0.4 * inch,
                           doc.pagesize[1] - 0.38 * inch,
                           "CONFIDENTIAL — RECRUITING REPORT")
    # Footer
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, 0, doc.pagesize[0], 0.35 * inch, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(doc.pagesize[0] / 2, 0.13 * inch,
                             f"Generated by IceIQ Analytics  |  Page {doc.page}")
    canvas.restoreState()


def _build_player_page(player: PlayerReport, s: dict) -> list:
    """Return a list of ReportLab flowables for one player page."""
    story = []
    W = letter[0]

    # ── Player Card ──────────────────────────────────────────────────────────
    story.append(Paragraph(f"#{player.name}", s["title"]))
    story.append(Paragraph(
        f"{player.team}  ·  {player.position}  ·  Shoots {player.shot_hand}",
        s["subtitle"],
    ))
    story.append(Spacer(1, 0.12 * inch))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=ICE_BLUE, spaceAfter=6))

    # Key stats summary table
    stats_data = [
        ["TOI", "Avg Speed", "Max Speed", "Distance", "Sprints"],
        [
            player.toi_str,
            f"{player.avg_speed_kmh:.1f} km/h",
            f"{player.max_speed_kmh:.1f} km/h",
            f"{player.total_distance_m / 1000:.2f} km",
            str(player.sprint_count),
        ],
    ]
    col_w = [(W - inch) / 5] * 5
    stats_tbl = Table(stats_data, colWidths=col_w)
    stats_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("BACKGROUND",  (0, 1), (-1, 1), LIGHT_GREY),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",    (0, 1), (-1, 1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY]),
        ("GRID",        (0, 0), (-1, -1), 0.5, MID_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(stats_tbl)
    story.append(Spacer(1, 0.18 * inch))

    # ── Charts Row ───────────────────────────────────────────────────────────
    story.append(Paragraph("Performance Charts", s["section"]))

    speed_img = _speed_bar_chart(player)
    donut_img = _zone_donut(player)
    heat_img  = _ice_heatmap(player)

    chart_col_w = [(W - inch) / 3] * 3
    chart_tbl = Table([[speed_img, donut_img, heat_img]],
                       colWidths=chart_col_w)
    chart_tbl.setStyle(TableStyle([
        ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",    (0, 0), (-1, -1), 0.5, MID_GREY),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(chart_tbl)
    story.append(Spacer(1, 0.18 * inch))

    # ── Events Table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Event Summary", s["section"]))
    ev = player.events
    event_data = [
        ["Event",         "Count"],
        ["Zone Entries",  str(ev.get("zone_entries", 0))],
        ["Faceoffs",      str(ev.get("faceoffs", 0))],
        ["Sprints",       str(ev.get("sprints", 0))],
        ["Goal Area",     str(ev.get("goal_area", 0))],
    ]
    ev_col_w = [3.5 * inch, 1.5 * inch]
    ev_tbl = Table(event_data, colWidths=ev_col_w)
    ev_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (0, -1), "LEFT"),
        ("ALIGN",       (1, 0), (1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("GRID",        (0, 0), (-1, -1), 0.5, MID_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
    ]))
    story.append(ev_tbl)
    story.append(Spacer(1, 0.18 * inch))

    # ── Scout Notes ──────────────────────────────────────────────────────────
    story.append(Paragraph("Scout Notes", s["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=MID_GREY, spaceAfter=2))
    for _ in range(8):
        story.append(Paragraph("_" * 95, s["note_line"]))

    return story


# ── Public API ───────────────────────────────────────────────────────────────

def generate_report(player_data: PlayerReport, output_path: str) -> None:
    """Generate a single-player recruiting report PDF."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = BaseDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.7 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
    )

    content_w = letter[0] - inch
    content_h = letter[1] - 1.2 * inch
    frame = Frame(0.5 * inch, 0.5 * inch, content_w, content_h,
                  id="main", showBoundary=0)
    template = PageTemplate(id="main", frames=[frame],
                             onPage=_header_footer)
    doc.addPageTemplates([template])

    s = _styles()
    story = _build_player_page(player_data, s)
    doc.build(story)
    print(f"[recruiting_report] Saved → {output_path}")


def generate_team_report(players: list, output_dir: str) -> None:
    """Generate one PDF per player into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for player in players:
        safe_name = player.name.replace(" ", "_").replace("#", "")
        path = os.path.join(output_dir, f"{safe_name}_report.pdf")
        generate_report(player, path)


# ── Test / demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = PlayerReport(
        name="47 한승원",
        team="Aigis HC",
        position="Center",
        shot_hand="Right",
        avg_speed_kmh=18.4,
        max_speed_kmh=31.7,
        total_distance_m=5840,
        sprint_count=12,
        frame_count=32400,          # 18 min @ 30 fps
        zone_time={"def": 210, "neu": 380, "off": 490},
        events={
            "zone_entries": 14,
            "faceoffs": 22,
            "sprints": 12,
            "goal_area": 7,
        },
    )

    out = os.path.join(
        os.path.dirname(__file__), "..", "data", "reports", "test_report.pdf"
    )
    generate_report(sample, os.path.normpath(out))

    import subprocess
    subprocess.run(
        ["openclaw", "system", "event",
         "--text", "recruiting_report complete: PDF generated",
         "--mode", "now"],
        check=False,
    )
