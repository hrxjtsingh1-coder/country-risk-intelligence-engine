"""Public-facing Streamlit application for the Country Risk Intelligence Engine."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.commentary.generate_commentary import generate_report
from src.runtime.live_data import create_wide_panel_from_long, fetch_live_data, indicator_records
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel, top_drivers

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DEMO_PATH = ROOT / "data" / "demo" / "panel_wide.csv"
MODEL_VERSION = "1.3.0"

st.set_page_config(
    page_title="Country Risk Intelligence Engine",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=21600, show_spinner=False)
def load_live_bundle() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    long_panel, metadata = fetch_live_data(2012, pd.Timestamp.utcnow().year)
    wide_panel = create_wide_panel_from_long(long_panel)
    return long_panel, metadata, wide_panel


@st.cache_data(ttl=3600, show_spinner=False)
def load_demo_panel() -> pd.DataFrame:
    return pd.read_csv(DEMO_PATH)


def style() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#06080d;--panel:#0d121a;--line:rgba(156,171,190,.14);--text:#f5f8fb;--muted:#91a0b2;--cyan:#67e8f9;--violet:#a88bff;--green:#55d79b;--amber:#ffd66b;--orange:#ff9f62;--red:#ff7180}
html,body,[class*="css"]{font-family:Inter,system-ui,sans-serif}
.stApp{background:radial-gradient(circle at 3% 0%,rgba(103,232,249,.07),transparent 24%),radial-gradient(circle at 98% 4%,rgba(168,139,255,.08),transparent 27%),linear-gradient(180deg,#07090e,#05070b);color:var(--text);overflow-x:hidden}
.block-container{max-width:1480px;padding-top:1rem;padding-bottom:3.4rem;overflow-x:hidden}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c1119,#070a0f);border-right:1px solid var(--line)}
section[data-testid="stSidebar"]>div{padding-top:1rem}
.stButton>button,.stDownloadButton>button{border:1px solid var(--line);border-radius:10px;background:rgba(13,19,28,.9);color:var(--text);font-weight:600}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:rgba(103,232,249,.42);box-shadow:0 0 20px rgba(103,232,249,.08)}
.stSelectbox [data-baseweb="select"]>div,.stNumberInput input{background:rgba(13,19,28,.92);border-color:var(--line)}
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:24px;padding:28px 30px;background:radial-gradient(circle at 86% 20%,rgba(103,232,249,.11),transparent 29%),radial-gradient(circle at 12% 88%,rgba(168,139,255,.08),transparent 31%),linear-gradient(135deg,rgba(17,25,35,.97),rgba(8,12,18,.94));box-shadow:0 22px 75px rgba(0,0,0,.26);animation:rise .55s ease both}
.hero:after{content:"";position:absolute;width:300px;height:300px;right:-145px;top:-155px;border:1px solid rgba(103,232,249,.13);border-radius:50%;box-shadow:0 0 0 28px rgba(103,232,249,.018),0 0 0 58px rgba(168,139,255,.018);animation:orbit 12s linear infinite}
.kicker{font:500 10px JetBrains Mono,monospace;letter-spacing:.17em;color:var(--cyan);text-transform:uppercase}.hero h1{position:relative;z-index:1;margin:6px 0 10px;font-size:clamp(34px,5vw,59px);line-height:1;letter-spacing:-.055em}.hero h1 span{background:linear-gradient(95deg,#fff,#c5f7fa 44%,#b69cff);-webkit-background-clip:text;background-clip:text;color:transparent}.hero p{position:relative;z-index:1;max-width:900px;margin:0;color:var(--muted);font-size:14px;line-height:1.65}
.status{margin:13px 0;border:1px solid var(--line);border-radius:14px;padding:12px 15px;background:rgba(13,18,26,.84)}.status strong{font-size:11px;letter-spacing:.04em}.status small{display:block;margin-top:4px;color:var(--muted);line-height:1.45}.status.live{border-color:rgba(85,215,155,.28)}.status.live strong{color:var(--green)}.status.demo{border-color:rgba(255,214,107,.28)}.status.demo strong{color:var(--amber)}.status.bad{border-color:rgba(255,113,128,.3)}.status.bad strong{color:var(--red)}
.section{margin-top:25px}.section h2{margin:0;font-size:20px;letter-spacing:-.025em}.sub{margin:4px 0 12px;color:var(--muted);font-size:12px}.card{height:100%;border:1px solid var(--line);border-radius:17px;padding:17px;background:linear-gradient(145deg,rgba(17,25,35,.91),rgba(10,14,20,.88));box-shadow:0 14px 45px rgba(0,0,0,.17)}.eyebrow{font:500 10px JetBrains Mono,monospace;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}.big{margin-top:6px;font-size:36px;line-height:1.02;font-weight:800;letter-spacing:-.05em}.risk{margin-top:4px;font-size:21px;font-weight:800}.muted{font-size:11px;color:var(--muted)}
.callout{border:1px solid rgba(103,232,249,.17);border-radius:16px;padding:15px 16px;background:linear-gradient(135deg,rgba(103,232,249,.05),rgba(168,139,255,.04));color:#d9e2ec;font-size:13px;line-height:1.65}.callout strong{color:#fff}
.driver{display:flex;justify-content:space-between;gap:12px;padding:10px 0;border-bottom:1px solid rgba(156,171,190,.08)}.driver:last-child{border-bottom:0}.driver b{font-size:13px}.driver small{display:block;margin-top:3px;color:var(--muted)}.contrib{font:600 13px JetBrains Mono,monospace;white-space:nowrap}.footer{margin-top:28px;padding-top:15px;border-top:1px solid var(--line);color:#657286;font-size:10px;line-height:1.6}
div[data-testid="stTabs"] button{font-weight:600}.dataframe{border-radius:12px}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}@keyframes orbit{to{transform:rotate(360deg)}}
@media(max-width:700px){.block-container{padding-left:.65rem;padding-right:.65rem}.hero{padding:22px 18px;border-radius:18px}.hero h1{font-size:30px}.hero p{font-size:13px}.section{margin-top:20px}.section h2{font-size:18px}.card{padding:14px}.big{font-size:31px}.status{padding:10px 12px}}
</style>
        """,
        unsafe_allow_html=True,
    )


def chart_config() -> dict[str, Any]:
    return {
        "displayModeBar": False,
        "scrollZoom": False,
        "doubleClick": False,
        "responsive": True,
    }


def lock_chart(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=26, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#91a0b2"),
        showlegend=False,
        dragmode=False,
    )
    fig.update_xaxes(fixedrange=True, showgrid=False, zeroline=False)
    fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="rgba(156,171,190,.06)", zeroline=False)
    return fig


def configs() -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, list[str]]]:
    countries = yaml.safe_load((CONFIG_DIR / "countries.yaml").read_text(encoding="utf-8")) or {}
    names = {str(x["iso3"]): str(x["name"]) for x in countries.get("countries", []) if x.get("iso3")}
    peers = {str(k): [str(v) for v in values] for k, values in countries.get("peer_groups", {}).items()}
    indicators = {str(x["code"]): x for x in indicator_records()}
    return names, indicators, peers


def render_unavailable(error: str) -> None:
    st.markdown('<div class="status bad"><strong>● LIVE DATA UNAVAILABLE</strong><small>The official public data path could not be verified. Synthetic observations are not being substituted silently.</small></div>', unsafe_allow_html=True)
    with st.expander("Technical details"):
        st.code(error or "Unknown error")
    a, b = st.columns(2)
    with a:
        if st.button("↻ Retry live data", type="primary", use_container_width=True):
            load_live_bundle.clear(); st.session_state.panel = None; st.session_state.long = None; st.session_state.meta = None; st.session_state.error = None; st.rerun()
    with b:
        if st.button("Open Demo Dataset", use_container_width=True):
            st.session_state.mode = "demo"; st.session_state.panel = load_demo_panel(); st.session_state.long = None; st.session_state.meta = None; st.session_state.error = None; st.rerun()
    st.markdown('<div class="callout"><strong>Demo mode remains fully interactive.</strong><br>The same score, driver, peer and scenario engine is used against the tracked synthetic fixture.</div>', unsafe_allow_html=True)


def load_state() -> None:
    st.session_state.setdefault("mode", "live")
    st.session_state.setdefault("panel", None)
    st.session_state.setdefault("long", None)
    st.session_state.setdefault("meta", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("country", "IND")
    st.session_state.setdefault("year", None)
    st.session_state.setdefault("peer_group", "Global")
    st.session_state.setdefault("scenario", None)
    if st.session_state.mode == "live" and st.session_state.panel is None:
        try:
            long, meta, wide = load_live_bundle()
            if wide.empty:
                raise RuntimeError("The official public sources returned no usable observations.")
            st.session_state.long = long
            st.session_state.meta = meta
            st.session_state.panel = wide
            st.session_state.error = None
        except Exception as exc:
            st.session_state.error = str(exc)
    elif st.session_state.mode == "demo" and st.session_state.panel is None:
        st.session_state.panel = load_demo_panel()


def main() -> None:
    style(); load_state(); names, ind, peer_groups = configs(); panel = st.session_state.panel
    st.markdown('<div class="hero"><div class="kicker">Global macro · country risk intelligence</div><h1>Country Risk <span>Intelligence Engine</span></h1><p>Explore relative country-risk positioning, understand the drivers behind the score, compare peers, and run transparent historical-sensitivity scenarios using public macroeconomic data.</p></div>', unsafe_allow_html=True)

    live = st.session_state.mode == "live" and panel is not None and not panel.empty
    meta = st.session_state.meta or {}
    if not live and st.session_state.mode != "demo":
        render_unavailable(st.session_state.error or "Live data was not available."); return

    if live:
        q = meta.get("quality", {})
        retrieved = meta.get("retrieved_at")
        try: when = pd.Timestamp(retrieved).strftime("%d %b %Y · %H:%M UTC") if retrieved else "—"
        except Exception: when = str(retrieved or "—")
        latest = meta.get("latest_valid_analysis_year") or meta.get("latest_available_observation") or "—"
        st.markdown(f'<div class="status live"><strong>● LIVE PUBLIC DATA</strong><small>World Bank Indicators API + FRED US-only enrichment · Retrieved {when} · Latest valid analysis year: {latest} · Valid observations: {float(q.get("coverage",0))*100:.1f}%</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status demo"><strong>● DEMO DATA · SYNTHETIC DATASET</strong><small>Tracked deterministic fixture · interface / methodology showcase only · not live public data</small></div>', unsafe_allow_html=True)

    available = sorted(panel["country_iso3"].dropna().astype(str).str.upper().unique())
    if not available:
        render_unavailable("No countries were available after collection."); return
    default_country = "IND" if "IND" in available else available[0]
    years = sorted(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).unique())
    default_year = int(meta.get("latest_valid_analysis_year") or max(years)) if years else 2020

    with st.sidebar:
        st.markdown("## Research controls")
        country = st.selectbox("Country", available, index=available.index(st.session_state.country) if st.session_state.country in available else available.index(default_country), format_func=lambda x: names.get(x, x), key="country")
        year = st.selectbox("Analysis year", years, index=years.index(st.session_state.year) if st.session_state.year in years else len(years)-1, key="year")
        peer = st.selectbox("Comparison set", ["Global", "Advanced", "Emerging"], index=["Global","Advanced","Emerging"].index(st.session_state.peer_group) if st.session_state.peer_group in ["Global","Advanced","Emerging"] else 0, key="peer_group")
        st.divider()
        if live:
            if st.button("↻ Refresh live data", use_container_width=True):
                load_live_bundle.clear(); st.session_state.panel=None; st.session_state.long=None; st.session_state.meta=None; st.session_state.scenario=None; st.rerun()
            if st.button("Open demo dataset", use_container_width=True):
                st.session_state.mode="demo"; st.session_state.panel=load_demo_panel(); st.session_state.meta=None; st.session_state.long=None; st.session_state.scenario=None; st.rerun()
        else:
            if st.button("Return to live data", type="primary", use_container_width=True):
                st.session_state.mode="live"; st.session_state.panel=None; st.session_state.long=None; st.session_state.meta=None; st.session_state.error=None; load_live_bundle.clear(); st.rerun()
        st.caption(f"Model {MODEL_VERSION}")
        st.caption("Charts use hover/animation; zoom and pan are disabled.")

    scores, drivers = score_panel(panel)
    current = scores[(scores.country_iso3 == country) & (scores.year == year)]
    if current.empty or pd.isna(current.iloc[0]["risk_score"]):
        st.warning("Insufficient data to score this country-year."); return
    row = current.iloc[0]; score=float(row.risk_score); band=str(row.risk_band); completeness=float(row.data_completeness)
    previous=scores[(scores.country_iso3==country)&(scores.year==year-1)]
    yoy=score-float(previous.iloc[0].risk_score) if not previous.empty and pd.notna(previous.iloc[0].risk_score) else math.nan

    if peer == "Advanced": peer_codes = peer_groups.get("advanced", [])
    elif peer == "Emerging": peer_codes = peer_groups.get("emerging", [])
    else: peer_codes = list(scores[scores.year == year].country_iso3.unique())
    peers=scores[(scores.year==year)&(scores.country_iso3.isin(peer_codes))].dropna(subset=["risk_score"])
    if not peers.empty:
        position=int((peers.risk_score<score).sum()+1); median=float(peers.risk_score.median())
    else: position=None; median=math.nan

    top=top_drivers(drivers,country,year,n=6)
    lead=str(top.iloc[0]["label"]) if not top.empty else "available indicators"
    direction="above" if not math.isnan(median) and score>median else "below" if not math.isnan(median) else "not comparable with"

    st.markdown('<div class="section"><h2>Executive view</h2><div class="sub">The decision layer: current position, movement, peers and evidence quality.</div></div>', unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    with a: st.markdown(f'<div class="card"><div class="eyebrow">Relative risk</div><div class="big">{score:.1f}</div><div class="risk">{band}</div><div class="muted">0–100 model position</div></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="card"><div class="eyebrow">Year-on-year</div><div class="big">{"—" if math.isnan(yoy) else f"{yoy:+.1f}"}</div><div class="muted">Score-point change vs {year-1}</div></div>',unsafe_allow_html=True)
    with c: st.markdown(f'<div class="card"><div class="eyebrow">Peer position</div><div class="big">{f"#{position} / {len(peers)}" if position else "—"}</div><div class="muted">Selected comparison set</div></div>',unsafe_allow_html=True)
    with d: st.markdown(f'<div class="card"><div class="eyebrow">Score completeness</div><div class="big">{completeness*100:.0f}%</div><div class="muted">Configured weight observed</div></div>',unsafe_allow_html=True)

    peer_sentence=f"It is {direction} the peer median of {median:.1f}." if not math.isnan(median) else "A peer median is unavailable for this comparison set."
    st.markdown(f'<div class="section"><div class="callout"><strong>What does this mean?</strong><br>{names.get(country,country)} is in the <strong>{band.lower()}</strong> relative-risk band for {year}. The largest current model signal by absolute contribution is <strong>{lead}</strong>. {peer_sentence}<br><br><span class="muted">Relative-risk positioning is a research signal, not a credit rating, probability of default, forecast or investment recommendation.</span></div></div>',unsafe_allow_html=True)

    # Global animated risk map
    st.markdown('<div class="section"><h2>Global risk pulse</h2><div class="sub">Watch the cross-sectional risk landscape change through recent years. Hover a country for its model score.</div></div>',unsafe_allow_html=True)
    map_df=scores.dropna(subset=["risk_score"]).copy(); map_df["country_name"]=map_df.country_iso3.map(names).fillna(map_df.country_iso3); recent=sorted(map_df.year.unique())[-8:]
    scale=[[0,"#39b79b"],[.35,"#a4d98b"],[.58,"#ffd66b"],[.78,"#ff9d61"],[1,"#ff7080"]]
    last=map_df[map_df.year==recent[-1]]
    fig=go.Figure()
    for y in recent:
        f=map_df[map_df.year==y]
        fig.add_trace(go.Choropleth(locations=f.country_iso3,z=f.risk_score,text=f.country_name,locationmode="ISO-3",colorscale=scale,zmin=0,zmax=100,marker_line_width=.25,marker_line_color="#2a3542",colorbar=dict(title="Risk",len=.65),hovertemplate="%{text}<br>Relative risk: %{z:.1f}<extra></extra>",visible=(y==recent[-1])))
    steps=[]
    for i,y in enumerate(recent): steps.append(dict(label=str(int(y)),method="update",args=[{"visible":[j==i for j in range(len(recent))]}, {"title":{"text":f"Global relative-risk pulse · {int(y)}"}}]))
    fig.update_layout(height=520,margin=dict(l=0,r=0,t=30,b=0),paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#91a0b2",family="Inter"),geo=dict(scope="world",showframe=False,bgcolor="rgba(0,0,0,0)",landcolor="#121923",showcoastlines=True,coastlinecolor="rgba(145,160,179,.2)",projection_type="natural earth"),dragmode=False,title=dict(text=f"Global relative-risk pulse · {int(recent[-1])}",x=.01,font=dict(size=13,color="#dfe8f1")),sliders=[dict(active=len(recent)-1,currentvalue=dict(prefix="Year · "),steps=steps)])
    st.plotly_chart(fig,config=chart_config(),use_container_width=True)

    # Movers
    st.markdown('<div class="section"><h2>Risk movers</h2><div class="sub">Largest year-over-year score changes in the selected analysis panel.</div></div>',unsafe_allow_html=True)
    movers=scores.sort_values(["country_iso3","year"]).copy(); movers["yoy_change"]=movers.groupby("country_iso3")["risk_score"].diff(); movers=movers[movers.year==year].dropna(subset=["yoy_change"]).copy(); movers["Country"]=movers.country_iso3.map(names).fillna(movers.country_iso3)
    if not movers.empty:
        up=movers.nlargest(5,"yoy_change")[['Country','yoy_change']].copy(); down=movers.nsmallest(5,'yoy_change')[['Country','yoy_change']].copy(); m1,m2=st.columns(2)
        with m1:
            st.markdown('<div class="card"><div class="eyebrow">Highest deterioration</div>'+''.join(f'<div class="driver"><div><b>{r.Country}</b><small>YoY score change</small></div><div class="contrib" style="color:#ff7180">{r.yoy_change:+.1f}</div></div>' for r in up.itertuples())+'</div>',unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="card"><div class="eyebrow">Largest improvement</div>'+''.join(f'<div class="driver"><div><b>{r.Country}</b><small>YoY score change</small></div><div class="contrib" style="color:#55d79b">{r.yoy_change:+.1f}</div></div>' for r in down.itertuples())+'</div>',unsafe_allow_html=True)

    left,right=st.columns([1.02,.98])
    with left:
        st.markdown('<div class="section"><h2>Why is the score here?</h2><div class="sub">Largest contributions, translated from technical codes into analyst language.</div></div>',unsafe_allow_html=True)
        if top.empty: st.info("No driver decomposition is available for this slice.")
        else:
            for _,x in top.iterrows():
                value="—" if pd.isna(x.raw_value) else f"{float(x.raw_value):,.2f} {ind.get(str(x.indicator_code),{}).get('unit','')}".strip(); c="#ff7180" if float(x.weighted_contribution)>0 else "#55d79b"
                st.markdown(f'<div class="driver"><div><b>{x["label"]}</b><small>{value} · weight {float(x.weight)*100:.0f}% · technical code {x["indicator_code"]}</small></div><div class="contrib" style="color:{c}">{float(x.weighted_contribution):+.2f}</div></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section"><h2>Peer context</h2><div class="sub">Relative position inside the selected comparison set.</div></div>',unsafe_allow_html=True)
        p=peers.copy(); p["name"]=p.country_iso3.map(names).fillna(p.country_iso3); p=p.sort_values("risk_score").tail(12)
        if not p.empty:
            f=px.bar(p,x="risk_score",y="name",orientation="h",labels={"risk_score":"Relative risk","name":"Country"}); f.update_traces(marker_color="#8870d2",hovertemplate="%{y}<br>%{x:.1f}<extra></extra>"); lock_chart(f,360); f.update_xaxes(range=[0,100]); st.plotly_chart(f,config=chart_config(),use_container_width=True)

    st.markdown('<div class="section"><h2>Context over time</h2><div class="sub">Historical relative positioning. Zoom and pan are intentionally disabled.</div></div>',unsafe_allow_html=True)
    hist=scores[scores.country_iso3==country].sort_values("year"); f=px.line(hist,x="year",y="risk_score",markers=True,labels={"year":"Year","risk_score":"Relative risk"}); f.update_traces(line=dict(color="#67e8f9",width=2.5),marker=dict(size=5),hovertemplate="Year: %{x}<br>Relative risk: %{y:.1f}<extra></extra>"); lock_chart(f,360); f.update_yaxes(range=[0,100]); st.plotly_chart(f,config=chart_config(),use_container_width=True)

    st.markdown('<div class="section"><h2>Scenario laboratory</h2><div class="sub">Run one hypothetical shock and inspect the model’s historical sensitivity. This is not a forecast.</div></div>',unsafe_allow_html=True)
    candidates=[r for r in indicator_records() if float(r.get("weight",0))>0 and r.get("code") in panel.columns]
    options=[(r["code"],r.get("label",r["code"]),r.get("unit","") or "units") for r in candidates]
    if options:
        driver_code=st.selectbox("Shock driver",options,format_func=lambda z:f"{z[1]} · {z[2]}",key="scenario_driver"); amount=st.number_input("Shock amount",value=1.0,step=.5,help=f"Change in {driver_code[2]} used by the historical sensitivity model.",key="scenario_amount")
        if st.button("Run historical sensitivity",type="primary"):
            targets=[code for code,_,_ in options if code!=driver_code[0]][:3]
            try: st.session_state.scenario=run_shock_scenario(panel,country,year,driver_code[0],float(amount),targets)
            except Exception as exc: st.session_state.scenario={"error":str(exc)}
        sc=st.session_state.get("scenario")
        if sc:
            if sc.get("error"): st.error(str(sc["error"]))
            else:
                delta=float(sc.get("delta",0)); flag=" · OUT-OF-SAMPLE SHOCK" if sc.get("out_of_sample_shock") else ""; colour="#ff7180" if delta>0 else "#55d79b" if delta<0 else "#91a0b2"
                st.markdown(f'<div class="card"><div class="eyebrow">Historical sensitivity{flag}</div><div class="big">{float(sc["baseline_score"]):.1f} → {float(sc["scenario_score"]):.1f}</div><div class="risk" style="color:{colour}">{delta:+.1f} pts</div><div class="muted">{sc.get("information_assessment","UNKNOWN")} · {sc.get("model_specification","")}</div></div>',unsafe_allow_html=True)
                for x in sorted(sc.get("indicator_deltas",[]),key=lambda z:abs(float(z.get("estimated_delta",0))),reverse=True)[:3]: st.markdown(f'<div class="driver"><div><b>{ind.get(str(x.get("indicator_code")),{}).get("label",x.get("indicator_code"))}</b><small>R² {float(x.get("r_squared",0)):.2f} · n={int(x.get("n_obs",0))}</small></div><div class="contrib">{float(x.get("estimated_delta",0)):+.2f}</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="section"><h2>Evidence & methodology</h2><div class="sub">Everything needed to inspect the data, model and outputs.</div></div>',unsafe_allow_html=True)
    tab1,tab2,tab3,tab4=st.tabs(["Data quality","Indicator dictionary","Model card","Exports"])
    with tab1:
        q=meta.get("quality",{}) if live else {}; st.dataframe(pd.DataFrame([{"Check":"Countries","Value":int(q.get("countries",len(available)))},{"Check":"Indicators","Value":int(q.get("indicators",len(ind)))},{"Check":"Valid observations","Value":int(q.get("valid_observations",len(panel)))},{"Check":"Missing","Value":int(q.get("missing",0))},{"Check":"Out-of-range","Value":int(q.get("out_of_range",0))},{"Check":"Coverage","Value":f"{float(q.get('coverage',completeness))*100:.1f}%"}]),use_container_width=True,hide_index=True); st.caption(f"Latest observation: {meta.get('latest_available_observation','—')} · Latest valid common year: {meta.get('latest_valid_analysis_year',year)}")
    with tab2:
        rows=[]
        for x in indicator_records(): rows.append({"Indicator":x.get("label"),"Code":x.get("code"),"Source":x.get("source"),"Unit":x.get("unit"),"Weight":f"{float(x.get('weight',0))*100:.0f}%","Risk direction":"Higher = higher risk" if float(x.get('risk_direction',1))>0 else "Higher = lower risk","Transformation":x.get("transformation")})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    with tab3:
        st.markdown("**Purpose**  
Relative country-risk positioning from public macroeconomic indicators.\n\n**Score**  
Direction-adjusted weighted cross-sectional z-scores mapped around a 50-point midpoint and bounded to 0–100. Missing indicators are excluded and observed weights are renormalized.\n\n**Scenario**  
Pooled-panel bivariate OLS historical sensitivity. It has no structural controls, lags or causal identification.\n\n**Responsible use**  
This is a research prototype, not a credit rating, probability of default, forecast or investment recommendation.")
    with tab4:
        selected=panel[(panel.country_iso3==country)&(panel.year==year)]; st.download_button("Download country-year data",selected.to_csv(index=False).encode(),file_name=f"{country}_{year}_data.csv",mime="text/csv",use_container_width=True); st.download_button("Download all scores",scores.to_csv(index=False).encode(),file_name="country_risk_scores.csv",mime="text/csv",use_container_width=True)
        report=generate_report(names.get(country,country),country,int(year),scores,drivers,st.session_state.get("scenario"),peer_codes)
        st.download_button("Download analyst brief",report.encode(),file_name=f"{country}_{year}_analyst_brief.md",mime="text/markdown",use_container_width=True)

    st.markdown('<div class="section"><h2>Primary sources</h2><div class="sub">The public data providers behind the live collection path.</div></div>',unsafe_allow_html=True)
    s1,s2=st.columns(2)
    with s1: st.markdown('<div class="card"><div class="eyebrow">World Bank</div><div class="risk">Indicators API</div><div class="muted">Annual macro indicators · api.worldbank.org/v2/</div></div>',unsafe_allow_html=True)
    with s2: st.markdown('<div class="card"><div class="eyebrow">FRED</div><div class="risk">DFF · US only</div><div class="muted">Effective Federal Funds Rate enrichment · excluded from composite score.</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="footer">Country Risk Intelligence Engine · research / education prototype · model {MODEL_VERSION} · live public data is fetched at runtime when available · synthetic demo data is explicitly labelled.</div>',unsafe_allow_html=True)


if __name__ == "__main__":
    main()
