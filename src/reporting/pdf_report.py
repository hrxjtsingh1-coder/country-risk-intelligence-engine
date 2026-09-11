"""Institutional-style PDF country risk report.

Formats ONLY what the engine already computed — the PDF builder never
re-scores, re-collects, or re-runs a scenario. Callers assemble a
`CountryRiskReportData` from the same objects the dashboard already holds
(scores, drivers, pillar scores, scenario output, commentary text) and this
module lays them out as a one-click export. Because it works on plain data
objects it is also usable from a CLI or a future API without importing
Streamlit.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from io import BytesIO

from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm

INK = colors.HexColor("#e8eef6")
DIM = colors.HexColor("#9aa7b8")
FAINT = colors.HexColor("#6b7a90")
PANEL = colors.HexColor("#121a29")
PANEL_LINE = colors.HexColor("#243350")
CYAN = colors.HexColor("#5ee7f2")
ORANGE = colors.HexColor("#ff9f5b")
RED = colors.HexColor("#ff6b7a")
GREEN = colors.HexColor("#54d69a")


@dataclass
class PillarRow:
    name: str
    score: float | None
    band: str


@dataclass
class DriverRow:
    label: str
    value_text: str
    contribution_points: float


@dataclass
class ScenarioChannelRow:
    indicator: str
    estimated_delta: float
    r_squared: float
    n_obs: int


@dataclass
class ScenarioRow:
    name: str
    driver_code: str
    baseline_score: float
    scenario_score: float
    delta: float
    baseline_band: str
    scenario_band: str
    information: str
    narrative: str
    out_of_sample: bool = False
    channels: list[ScenarioChannelRow] = field(default_factory=list)


@dataclass
class CountryRiskReportData:
    """Everything the PDF needs, already computed by the caller."""

    country_label: str
    iso3: str
    year: int
    risk_score: float
    band: str
    coverage_value: float | None  # 0..1 fraction
    trend_direction: str  # Deteriorating / Improving / Stable / ""
    trend_1y_delta: float | None
    report_text: str  # the existing analyst commentary, plain text
    sources: list[str]
    generated_at: str
    using_demo_data: bool
    data_verified: bool
    provenance_sources: str
    provenance_asof: str
    manifest_hash: str
    pillars: list[PillarRow]
    drivers: list[DriverRow]
    scenario: ScenarioRow | None = None
    trajectory_years: list[int] = field(default_factory=list)
    trajectory_scores: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


DIM_HEX = DIM.hexval()
CYAN_HEX = CYAN.hexval()
RED_HEX = RED.hexval()
ORANGE_HEX = ORANGE.hexval()
GREEN_HEX = GREEN.hexval()
INK_HEX = INK.hexval()


def _esc(text: object) -> str:
    return html.escape(str(text))


def _dim(text: str) -> str:
    return f'<font color="{DIM_HEX}">{text}</font>'


def _km_text(content: str) -> Paragraph:
    """Nickname in the running footer of the report."""
    return Paragraph(content, ParagraphStyle(name="micro", fontSize=6.5, textColor=DIM, leading=8))


# ---------------------------------------------------------------------------
# Trajectory chart (drawn natively so no headless renderer is needed)
# ---------------------------------------------------------------------------


def _trajectory_drawing(years: list[int], scores: list[float]) -> Drawing:
    """A 0-100 score trajectory line with the band reference lines."""
    plot_w, plot_h = 470, 150
    x0, y0 = 40, 22
    draw = Drawing(plot_w + 60, plot_h + 40)

    draw.add(Rect(x0 - 6, y0 - 6, plot_w + 12, plot_h + 12, fillColor=PANEL, strokeColor=PANEL_LINE, strokeWidth=0.5))

    grid = [40, 60, 80]
    grid_color = colors.HexColor("#3a4a63")
    shade = colors.HexColor("#2a1c22")
    for level in (80,):
        y = y0 + plot_h * level / 100.0
        draw.add(Rect(x0, y, plot_w, y0 + plot_h - y, fillColor=shade, strokeWidth=0))

    for level in grid:
        y = y0 + plot_h * level / 100.0
        draw.add(Line(x0, y, x0 + plot_w, y, strokeColor=grid_color, strokeWidth=0.5))
        draw.add(String(x0 - 6, y - 3, str(level), fontSize=6, fillColor=FAINT))

    if len(years) != len(scores) or len(scores) < 2:
        draw.add(String(x0 + 20, y0 + plot_h / 2, "No scored trajectory available.", fontSize=8, fillColor=DIM))
        return draw

    lo, hi = min(years), max(years)
    xspan = (hi - lo) or 1
    pts = []
    for year, score in zip(years, scores, strict=False):
        x = x0 + plot_w * (year - lo) / xspan
        y = y0 + plot_h * max(0.0, min(100.0, float(score))) / 100.0
        pts.append((x, y))

    draw.add(PolyLine(pts, strokeColor=CYAN, strokeWidth=1.6))
    for x, y in pts:
        draw.add(Circle(x, y, 2.2, fillColor=CYAN, strokeColor=PANEL, strokeWidth=0.6))

    for year in years:
        x = x0 + plot_w * (year - lo) / xspan
        draw.add(String(x - 6, y0 - 12, str(int(year)), fontSize=6, fillColor=FAINT))
    return draw


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def _cover_flow(data: CountryRiskReportData):
    flows = []

    flows.append(Paragraph(f'<font color="{RED_HEX}">COUNTRY RISK INTELLIGENCE ENGINE</font>', _title()))
    flows.append(
        Paragraph(
            f"{_esc(data.country_label)} — Institutional Risk Note",
            ParagraphStyle(name="coverTitle", fontName="Helvetica-Bold", fontSize=26, leading=32, textColor=INK),
        )
    )
    flows.append(
        Paragraph(
            f"{_esc(data.iso3)} &nbsp;·&nbsp; {data.year} &nbsp;·&nbsp; composite score "
            f'<font color="{CYAN_HEX}"><b>{_esc(data.risk_score)}</b></font> / 100 '
            f"&nbsp;·&nbsp; {_esc(data.band)} risk band",
            ParagraphStyle(name="coverSub", fontSize=11, leading=16, textColor=DIM),
        )
    )
    flows.append(Spacer(1, 14))
    flows.append(HRFlowable(width="100%", thickness=0.7, color=PANEL_LINE))

    if data.using_demo_data:
        flows.append(
            Paragraph(
                f'<font color="{ORANGE_HEX}"><b>SYNTHETIC DEMO DATA</b></font> — these numbers are '
                "interface/methodology demonstrations, not real economic observations.",
                ParagraphStyle(name="demo", fontSize=9, leading=13, textColor=DIM, spaceBefore=8),
            )
        )
    return flows


def _snapshot_table(data: CountryRiskReportData):
    def _cell(value: str, color: str = INK_HEX) -> Paragraph:
        return Paragraph(
            f"<b>{_esc(value)}</b>",
            ParagraphStyle(name="snap", fontSize=13, leading=16, textColor=colors.HexColor(color)),
        )

    def _label(label: str) -> Paragraph:
        return Paragraph(_dim(label), ParagraphStyle(name="snapL", fontSize=7.5, leading=10))

    coverage = "—" if data.coverage_value is None or math.isnan(data.coverage_value) else f"{data.coverage_value:.0%}"
    trend = data.trend_direction if data.trend_direction else "—"
    if data.trend_1y_delta is not None:
        trend_delta = f"{data.trend_1y_delta:+.1f} pts"
    else:
        trend_delta = "—"

    rows = [
        [
            _label("COMPOSITE RISK"),
            _label("RISK BAND"),
            _label("DATA COVERAGE"),
            _label("TREND (YoY)"),
        ],
        [
            _cell(str(data.risk_score), color=CYAN_HEX),
            _cell(
                data.band,
                color=(
                    GREEN if data.band in ("Low", "Moderate") else ORANGE if data.band == "Elevated" else RED
                ).hexval(),
            ),
            _cell(coverage),
            _cell(f"{trend} {trend_delta}"),
        ],
    ]
    table = Table(rows, colWidths=[118 * mm, 118 * mm, 118 * mm, 118 * mm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
            ]
        )
    )
    return table


def _analyst_flow(data: CountryRiskReportData):
    body = html.escape(str(data.report_text))
    body = body.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")
    style = ParagraphStyle(
        name="analyst",
        fontName="Helvetica",
        fontSize=8.8,
        leading=13.5,
        textColor=colors.HexColor("#d3dce8"),
        alignment=TA_LEFT,
    )
    return [
        Paragraph(_esc("ANALYST VIEW"), _kicker_style()),
        Paragraph(body, style),
    ]


def _pillar_table(data: CountryRiskReportData) -> Table:
    header = [_esc("PILLAR"), _esc("SCORE"), _esc("BAND")]
    rows = [header]
    for p in data.pillars:
        score = "—" if p.score is None else f"{p.score:.0f}"
        rows.append([_esc(p.name), _esc(score), _esc(p.band)])
    table = Table(rows, colWidths=[200 * mm, 70 * mm, 70 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTCOLOR", (0, 0), (-1, -1), INK),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("BACKGROUND", (0, 0), (-1, 0), PANEL),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PANEL, colors.HexColor("#0d1522")]),
                ("GRID", (0, 0), (-1, -1), 0.4, PANEL_LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _drivers_flow(data: CountryRiskReportData):
    flows = [Paragraph(_esc("MAIN DRIVERS"), _kicker_style())]
    if not data.drivers:
        flows.append(
            Paragraph(
                _dim("Not enough indicator coverage to enumerate the top drivers for this slice."),
                ParagraphStyle(name="p", fontSize=8.8, leading=13, textColor=INK),
            )
        )
        return flows
    for d in data.drivers:
        flows.append(
            Paragraph(
                f"<b>{_esc(d.label)}</b> &nbsp; {_dim(_esc(d.value_text))} &nbsp; "
                f'<font color="{CYAN_HEX}">{d.contribution_points:+.1f}</font> pts',
                ParagraphStyle(name="drv", fontSize=8.8, leading=13.5, textColor=INK, leftIndent=6),
            )
        )
    return flows


def _scenario_flow(data: CountryRiskReportData):
    sc = data.scenario
    flows = [Paragraph(_esc("SCENARIO OUTPUT"), _kicker_style())]
    if sc is None:
        flows.append(
            Paragraph(
                _dim("No scenario was run for this slice."),
                ParagraphStyle(name="p", fontSize=8.8, leading=13, textColor=INK),
            )
        )
        return flows

    flows.append(
        Paragraph(
            f"<b>{_esc(sc.name)}</b> &nbsp;·&nbsp; {_dim(_esc(sc.driver_code))} &nbsp;·&nbsp; "
            f"{_esc(sc.baseline_score)} → {_esc(sc.scenario_score)} "
            f"({sc.delta:+.1f} pts, {_esc(sc.scenario_band)}) &nbsp;·&nbsp; {_esc(sc.information)}"
            + (" &nbsp;·&nbsp; OUT-OF-SAMPLE SHOCK" if sc.out_of_sample else ""),
            ParagraphStyle(name="scn", fontSize=8.8, leading=13, textColor=INK),
        )
    )
    body = html.escape(str(sc.narrative)).replace("\n", "<br/>")
    flows.append(Paragraph(body, ParagraphStyle(name="scnB", fontSize=8.2, leading=12.5, textColor=DIM, spaceBefore=3)))

    if sc.channels:
        ch_rows = [[_esc("CHANNEL"), _esc("EST. DELTA"), _esc("R²"), _esc("N")]]
        for ch in sc.channels:
            ch_rows.append([_esc(ch.indicator), f"{ch.estimated_delta:+.2f}", f"{ch.r_squared:.2f}", _esc(ch.n_obs)])
        table = Table(ch_rows, colWidths=[200 * mm, 60 * mm, 50 * mm, 40 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("FONTCOLOR", (0, 0), (-1, -1), INK),
                    ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                    ("GRID", (0, 0), (-1, -1), 0.4, PANEL_LINE),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        flows.append(Spacer(1, 6))
        flows.append(table)
    return flows


def _sources_flow(data: CountryRiskReportData):
    flows = [Paragraph(_esc("SOURCES & PROVENANCE"), _kicker_style())]
    sources = [s for s in data.sources if s]
    if sources:
        for s in sources:
            flows.append(
                Paragraph(
                    f"&nbsp;&nbsp;•&nbsp;&nbsp;{_esc(s)}",
                    ParagraphStyle(name="src", fontSize=8, leading=12, textColor=DIM),
                )
            )
    else:
        flows.append(Paragraph(_dim("See config/indicators.yaml for the per-indicator source mapping."), _src_style()))

    provenance_bits = []
    if data.using_demo_data:
        provenance_bits.append(_dim("DEMO DATA — SYNTHETIC, not verified"))
    elif data.data_verified:
        provenance_bits.append(_dim("Verified public data"))
    else:
        provenance_bits.append(_dim("Provenance unavailable"))
    if data.provenance_sources:
        provenance_bits.append(_dim(f"Source: {_esc(data.provenance_sources)}"))
    if data.provenance_asof:
        provenance_bits.append(_dim(f"As of: {_esc(data.provenance_asof)}"))
    if data.manifest_hash:
        provenance_bits.append(_dim(f"Manifest: {_esc(data.manifest_hash)}"))
    provenance_bits.append(_dim(f"Generated: {_esc(data.generated_at)}"))
    flows.append(Paragraph(" &nbsp;·&nbsp; ".join(provenance_bits), _src_style()))
    return flows


# ---------------------------------------------------------------------------
# Small style factories
# ---------------------------------------------------------------------------


def _title():
    return ParagraphStyle(name="kicker", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=RED, spaceAfter=2)


def _kicker_style():
    return ParagraphStyle(
        name="k", fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=CYAN, spaceBefore=12, spaceAfter=4
    )


def _src_style():
    return ParagraphStyle(name="s", fontSize=8, leading=12.5, textColor=DIM, spaceBefore=2)


def _report_header_footer(canvas, doc) -> None:  # noqa: ANN001 - reportlab callback
    canvas.saveState()
    canvas.setStrokeColor(PANEL_LINE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - 24 * mm, PAGE_W - MARGIN, PAGE_H - 24 * mm)
    runner = _km_text("COUNTRY RISK INTELLIGENCE ENGINE")
    runner.wrap(200 * mm, 8)
    runner.drawOn(canvas, MARGIN, PAGE_H - 18 * mm)
    canvas.setFont("Helvetica", 6.5)
    canvas.setFillColor(DIM)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 18 * mm, "INSTITUTIONAL NOTE")
    canvas.restoreState()


def build_risk_report_pdf(data: CountryRiskReportData) -> bytes:
    """Return the PDF document bytes for one country-year risk note."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=28 * mm,
        bottomMargin=20 * mm,
        title=f"Country Risk — {data.country_label} ({data.iso3}) {data.year}",
        author="Country Risk Intelligence Engine",
    )

    flows: list = []
    flows.extend(_cover_flow(data))
    flows.append(Spacer(1, 10))
    flows.append(_snapshot_table(data))

    flows.append(Spacer(1, 16))
    flows.append(HRFlowable(width="100%", thickness=0.5, color=PANEL_LINE))
    flows.extend(_analyst_flow(data))

    if data.drivers:
        flows.extend(_drivers_flow(data))

    if data.pillars:
        flows.append(Paragraph(_esc("PILLAR BREAKDOWN"), _kicker_style()))
        flows.append(_pillar_table(data))

    if data.scenario is not None:
        flows.extend(_scenario_flow(data))

    if data.trajectory_years and data.trajectory_scores:
        flows.append(Paragraph(_esc("RISK TRAJECTORY"), _kicker_style()))
        flows.append(_trajectory_drawing(data.trajectory_years, data.trajectory_scores))
        flows.append(Spacer(1, 8))

    flows.append(Spacer(1, 12))
    flows.append(HRFlowable(width="100%", thickness=0.5, color=PANEL_LINE))
    flows.extend(_sources_flow(data))

    doc.build(flows, onFirstPage=_report_header_footer, onLaterPages=_report_header_footer)
    return buffer.getvalue()
