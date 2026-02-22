from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_pdf(run_dir: Path | str, out_pdf: Path | str, summary_text: str) -> None:
    """
    Create a PDF report from artifacts inside run_dir.
    Expected files (if available):
      - summary_table.json
      - figs/*.png
    """
    run_dir = Path(run_dir)
    out_pdf = Path(out_pdf)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

    story: List[Any] = []
    story.append(Paragraph("Readability Report", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(summary_text.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    # Try summary_table.json
    summary_path = run_dir / "summary_table.json"
    summary = _safe_read_json(summary_path)
    if isinstance(summary, dict) and summary:
        story.append(Paragraph("Key Metrics", styles["Heading2"]))
        rows = [["Metric", "Value"]]
        for k, v in summary.items():
            rows.append([str(k), str(v)])

        tbl = Table(rows, colWidths=[7*cm, 7*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

    # Add plots from figs/
        # Add plots from figs/
    figs_dir = run_dir / "figs"
    if figs_dir.exists():
        pngs = sorted(figs_dir.glob("*.png"))
        if pngs:
            story.append(Paragraph("Plots", styles["Heading2"]))
            story.append(Spacer(1, 0.2 * cm))

            max_w = 16.5 * cm          # etwas kleiner als Seitenbreite
            max_h = 22.5 * cm          # unterhalb A4 Höhe (mit Rändern/Überschrift)

            for p in pngs:
                story.append(Paragraph(p.name, styles["Italic"]))
                story.append(Spacer(1, 0.15 * cm))

                img = Image(str(p))
                iw, ih = img.imageWidth, img.imageHeight

                # scale-to-fit (preserve aspect ratio)
                scale = min(max_w / iw, max_h / ih, 1.0)
                img.drawWidth = iw * scale
                img.drawHeight = ih * scale

                story.append(img)
                story.append(Spacer(1, 0.5 * cm))
                story.append(PageBreak())

    doc.build(story)
