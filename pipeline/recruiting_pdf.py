"""
Recruiting PDF report generator for IceIQ.
Generates professional player scouting/recruiting reports as PDF.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime


def generate_recruiting_pdf(player_data: Dict[str, Any], output_path: str = "reports/recruiting_report.pdf") -> str:
    """
    Generate a recruiting PDF report for a player.

    Args:
        player_data: dict with keys:
            - name: str
            - jersey_number: int
            - team: str
            - position: str (forward/defense/goalie)
            - shot_hand: str (left/right)
            - metrics: dict of performance metrics
            - highlights: list of highlight timestamps
            - game_stats: dict of game statistics
        output_path: output PDF file path

    Returns:
        Path to generated PDF
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        return _generate_with_reportlab(player_data, output_path)
    except ImportError:
        return _generate_text_fallback(player_data, output_path.replace(".pdf", ".txt"))


def _generate_with_reportlab(player_data: Dict[str, Any], output_path: str) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor("#1a1a2e")
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"],
        fontSize=13, spaceAfter=4, textColor=colors.HexColor("#16213e"),
        spaceBefore=12
    )
    body_style = styles["BodyText"]

    story = []

    # Header
    story.append(Paragraph("IceIQ Player Scouting Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", body_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.2 * inch))

    # Player Identity
    story.append(Paragraph("Player Information", heading_style))
    identity_data = [
        ["Name", player_data.get("name", "Unknown")],
        ["Jersey #", str(player_data.get("jersey_number", "—"))],
        ["Team", player_data.get("team", "—")],
        ["Position", player_data.get("position", "—").capitalize()],
        ["Shot Hand", player_data.get("shot_hand", "—").capitalize()],
    ]
    identity_table = Table(identity_data, colWidths=[2 * inch, 4 * inch])
    identity_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f4f8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(identity_table)
    story.append(Spacer(1, 0.15 * inch))

    # Performance Metrics
    metrics = player_data.get("metrics", {})
    if metrics:
        story.append(Paragraph("Performance Metrics", heading_style))
        metrics_data = [["Metric", "Value", "Rating"]]
        for key, val in metrics.items():
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                display = f"{val:.3f}"
                rating = "★★★★★" if val > 0.8 else "★★★★" if val > 0.6 else "★★★" if val > 0.4 else "★★"
            else:
                display = str(val)
                rating = "—"
            metrics_data.append([label, display, rating])

        metrics_table = Table(metrics_data, colWidths=[2.5 * inch, 2 * inch, 1.5 * inch])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.15 * inch))

    # Game Stats
    game_stats = player_data.get("game_stats", {})
    if game_stats:
        story.append(Paragraph("Game Statistics", heading_style))
        stats_data = [[k.replace("_", " ").title(), str(v)] for k, v in game_stats.items()]
        stats_table = Table(stats_data, colWidths=[3 * inch, 3 * inch])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f4f8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 0.15 * inch))

    # Highlights
    highlights = player_data.get("highlights", [])
    if highlights:
        story.append(Paragraph("Highlight Moments", heading_style))
        for i, h in enumerate(highlights[:10], 1):
            if isinstance(h, dict):
                ts = h.get("timestamp", 0)
                desc = h.get("description", "Notable play")
                score = h.get("score", 0)
                mins, secs = divmod(int(ts), 60)
                story.append(Paragraph(
                    f"{i}. [{mins:02d}:{secs:02d}] {desc} (score: {score:.2f})",
                    body_style
                ))
            else:
                mins, secs = divmod(int(h), 60)
                story.append(Paragraph(f"{i}. [{mins:02d}:{secs:02d}]", body_style))

    # Footer note
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        "Generated by IceIQ — AI-powered ice hockey video analysis platform.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    return output_path


def _generate_text_fallback(player_data: Dict[str, Any], output_path: str) -> str:
    """Fallback text report when reportlab is not available."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = [
        "=" * 50,
        "IceIQ PLAYER SCOUTING REPORT",
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        "=" * 50,
        "",
        f"Name:         {player_data.get('name', 'Unknown')}",
        f"Jersey #:     {player_data.get('jersey_number', '—')}",
        f"Team:         {player_data.get('team', '—')}",
        f"Position:     {player_data.get('position', '—')}",
        f"Shot Hand:    {player_data.get('shot_hand', '—')}",
        "",
    ]
    metrics = player_data.get("metrics", {})
    if metrics:
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 30)
        for k, v in metrics.items():
            lines.append(f"  {k.replace('_', ' ').title()}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def batch_generate_reports(players: List[Dict[str, Any]], output_dir: str = "reports") -> List[str]:
    """Generate recruiting PDFs for multiple players."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for player in players:
        name = player.get("name", "player").replace(" ", "_").lower()
        jersey = player.get("jersey_number", "0")
        filename = f"{output_dir}/recruit_{name}_{jersey}.pdf"
        path = generate_recruiting_pdf(player, filename)
        paths.append(path)
    return paths


if __name__ == "__main__":
    # Test report generation
    test_player = {
        "name": "Alex Johnson",
        "jersey_number": 17,
        "team": "home",
        "position": "forward",
        "shot_hand": "left",
        "metrics": {
            "speed_score": 0.82,
            "shot_accuracy": 0.74,
            "passing_accuracy": 0.68,
            "defensive_coverage": 0.61,
            "possession_time": 0.77,
        },
        "game_stats": {
            "goals": 2,
            "assists": 1,
            "shots_on_goal": 5,
            "ice_time_minutes": 18,
        },
        "highlights": [
            {"timestamp": 342, "description": "Breakaway goal", "score": 0.95},
            {"timestamp": 1205, "description": "Defensive stop", "score": 0.78},
        ],
    }
    out = generate_recruiting_pdf(test_player, "/tmp/test_recruit.pdf")
    print(f"Report generated: {out}")
