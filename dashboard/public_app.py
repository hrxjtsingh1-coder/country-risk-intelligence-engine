"""Public-facing Country Risk Intelligence Engine."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.commentary.generate_commentary import generate_report
from src.runtime.live_data import create_wide_panel_from_long, data_quality, fetch_live_data, indicator_records
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel, top_drivers

ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "data" / "demo" / "panel_wide.csv"
CONFIG_PATH = ROOT / "config"
MODEL_VERSION = "1.3.0"

st.set_page_config(page_title="Country Risk Intelligence Engine", page_icon="◎", layout="wide", initial_sidebar_state="expanded")

@st.cache_data(ttl=21600, show_spinner=False)
def get_live_data():
    return fetch_live_data(2012, pd.Timestamp.utcnow().year)

@st.cache_data(ttl=3600, show_spinner=False)
def get_demo_data():
    return pd.read_csv(DEMO_PATH)


def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    :root{--bg:#07090e;--panel:#0d121a;--line:rgba(163,178,196,.14);--text:#f5f8fb;--muted:#91a0b2;--cyan:#67e8f9;--violet:#ad8bff;--green:#55d79b;--amber:#ffd66b;--orange:#ff9f62;--red:#ff7180}
    html,body,[class*="css"]{font-family:Inter,system-ui,sans-serif}
    .stApp{background:radial-gradient(circle at 5% 0%,rgba(103,232,249,.07),transparent 25%),radial-gradient(circle at 95% 5%,rgba(173,139,255,.08),transparent 26%),linear-gradient(180deg,#07090e,#05070b);color:var(--text);overflow-x:hidden}
    .block-container{max-width:1480px;padding-top:1rem;padding-bottom:3rem}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c1119,#070a0f);border-right:1px solid var(--line)}
    .hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:24px;padding:27px 30px;background:radial-gradient(circle at 86% 18%,rgba(103,232,249,.11),transparent 29%),radial-gradient(circle at 12% 90%,rgba(173,139,255,.08),transparent 30%),linear-gradient(135deg,rgba(17,25,35,.96),rgba(8,12,18,.94));box-shadow:0 22px 75px rgba(0,0,0,.28);animation:rise .55s ease both}
    .hero:after{content:"";position:absolute;width:300px;height:300px;right:-140px;top:-160px;border:1px solid rgba(103,232,249,.14);border-radius:50%;box-shadow:0 0 0 30px rgba(103,232,249,.02),0 0 0 62px rgba(173,139,255,.02);animation:orbit 11s linear infinite}
    .kicker{font:500 10px 'JetBrains Mono',monospace;letter-spacing:.18em;color:var(--cyan);text-transform:uppercase}
    .hero h1{position:relative;z-index:1;font-size:clamp(34px,5vw,60px);line-height:1;letter-spacing:-.05em;margin:7px 0 10px}.hero h1 span{background:linear-gradient(95deg,#fff,#c5f7fa 45%,#b69cff);-webkit-background-clip:text;background-clip:text;color:transparent}.hero p{position:relative;z-index:1;max-width:900px;color:var(--muted);line-height:1.65;margin:0;font-size:14px}
    .status{margin:13px 0;border:1px solid var(--line);border-radius:14px;padding:12px 15px;background:rgba(13,18,26,.82)}.status strong{font-size:11px;letter-spacing:.05em}.status small{display:block;color:var(--muted);margin-top:4px;line-height:1.45}.status.live{border-color:rgba(85,215,155,.27)}.status.live strong{color:var(--green)}.status.demo{border-color:rgba(255,214,107,.28)}.status.demo strong{color:var(--amber)}.status.bad{border-color:rgba(255,113,128,.3)}.status.bad strong{color:var(--red)}
    .section{margin-top:26px}.section h2{font-size:20px;line-height:1.2;margin:0;letter-spacing:-.02em}.sub{font-size:12px;color:var(--muted);margin:4px 0 13px}
    .card{height:100%;border:1px solid var(--line);border-radius:17px;background:linear-gradient(145deg,rgba(17,25,35,.9),rgba(10,14,20,.88));padding:17px;box-shadow:0 14px 45px rgba(0,0,0,.17)}.eyebrow{font:500 10px 'JetBrains Mono',monospace;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}.big{font-size:37px;line-height:1.05;font-weight:800;letter-spacing:-.045em;margin-top:7px}.risk{font-size:24px;font-weight:800;margin-top:4px}.muted{font-size:11px;color:var(--muted)}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 8px;font:500 10px 'JetBrains Mono',monospace;color:var(--muted)}
    .callout{border:1px solid rgba(103,232,249,.17);background:linear-gradient(135deg,rgba(103,232,249,.05),rgba(173,139,255,.04));border-radius:16px;padding:16px;line-height:1.65;color:#d9e2ec;font-size:13px}.callout strong{color:#fff}
    .driver{display:flex;justify-content:space-between;gap:12px;padding:11px 0;border-bottom:1px solid rgba(163,178,196,.08)}.driver:last-child{border-bottom:0}.driver b{font-size:13px}.driver small{display:block;color:var(--muted);margin-top:3px}.contrib{font:600 13px 'JetBrains Mono',monospace;white-space:nowrap}
    .footer{margin-top:32px;padding-top:16px;border-top:1px solid var(--line);color:#657286;font-size:10px;line-height:1.6}
    @keyframes rise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}@keyframes orbit{to{transform:rotate(360deg)}}
    @media(max-width:900px){.hero{padding:22px 19px}.hero h1{font-size:34px}.section{margin-top:21px}}
    @media(max-width:640px){.hero h1{font-size:29px}.big{font-size:33px}.section h2{font-size:18px}.card{padding:14px}.sub{font-size:11px}.status{padding:11px 12px}}
    </style>
    """, unsafe_allow_html=True)


def chart_config() -> dict:
    return {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "responsive": True}


def lock_chart(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8,r=8,t=26,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter",size=11,color="#91a0b2"), showlegend=False, dragmode=False)
    fig.update_xaxes(fixedrange=True, showgrid=False, zeroline=False)
    fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="rgba(163,178,196,.06)", zeroline=False)
    return fig


def cfg_maps() -> tuple[dict[str,str], dict[str,dict]]:
    raw=yaml.safe_load((CONFIG_PATH/"countries.yaml").read_text(encoding="utf-8")) or {}
    names={str(x.get("iso3")):str(x.get("name")) for x in raw.get("countries",[]) if x.get("iso3")}
    ind={str(x.get("code")):x for x in indicator_records()}
    return names,ind


def human(code: str, ind: dict[str,dict]) -> str:
    return str(ind.get(code,{}).get("label") or code.replace("_"," ").title())


def units(code: str, ind: dict[str,dict]) -> str:
    return str(ind.get(code,{}).get("unit") or "")


def load_initial_state() -> None:
    st.session_state.setdefault("mode","live")
    st.session_state.setdefault("panel",None)
    st.session_state.setdefault("meta",None)
    st.session_state.setdefault("long",None)
    st.session_state.setdefault("error",None)
    if st.session_state.mode=="live" and st.session_state.panel is None:
        try:
            long,meta=get_live_data(); wide=create_wide_panel_from_long(long)
            if wide.empty: raise RuntimeError("The official public sources returned no usable observations.")
            st.session_state.long=long; st.session_state.panel=wide; st.session_state.meta=meta; st.session_state.error=None
        except Exception as exc:
            st.session_state.error=str(exc)
    elif st.session_state.mode=="demo" and st.session_state.panel is None:
        st.session_state.panel=get_demo_data(); st.session_state.meta=None; st.session_state.long=None


def unavailable() -> None:
    st.markdown('<div class="status bad"><strong>LIVE DATA UNAVAILABLE</strong><small>The app could not verify official public data. Nothing synthetic has been substituted automatically.</small></div>',unsafe_allow_html=True)
    msg=st.session_state.get("error") or "Unknown live-data error."
    with st.expander("Technical details"):
        st.code(msg)
    a,b=st.columns(2)
    with a:
        if st.button("Retry live data",type="primary",use_container_width=True):
            get_live_data.clear(); st.session_state.panel=None; st.session_state.long=None; st.session_state.error=None; st.rerun()
    with b:
        if st.button("Open demo dataset",use_container_width=True):
            st.session_state.mode="demo"; st.session_state.panel=get_demo_data(); st.session_state.meta=None; st.session_state.long=None; st.session_state.error=None; st.rerun()
    st.markdown('<div class="callout"><strong>Demo mode is a complete working showcase.</strong><br>It uses the tracked synthetic panel with the same scoring, peer, driver and scenario logic.</div>',unsafe_allow_html=True)


def main() -> None:
    inject_css(); load_initial_state(); names,ind=cfg_maps()
    panel=st.session_state.panel
    st.markdown('<div class="hero"><div class="kicker">Global macro · risk intelligence</div><h1>Country Risk <span>Intelligence Engine</span></h1><p>A decision-support research interface for comparing countries, understanding the signals behind relative risk, and exploring transparent historical sensitivities.</p></div>',unsafe_allow_html=True)
    live=st.session_state.mode=="live" and panel is not None and not panel.empty
    meta=st.session_state.get("meta") or {}
    if not live and st.session_state.mode!="demo": unavailable(); return
    if live:
        q=meta.get("quality",{}); retrieved=meta.get("retrieved_at")
        when=pd.Timestamp(retrieved).strftime("%d %b %Y · %H:%M UTC") if retrieved else "—"
        latest=meta.get("latest_valid_analysis_year") or meta.get("latest_available_observation") or "—"
        st.markdown(f'<div class="status live"><strong>● LIVE PUBLIC DATA</strong><small>World Bank Indicators API + FRED US-only enrichment · Retrieved {when} · Latest valid analysis year: {latest} · Valid observation coverage: {float(q.get("coverage",0))*100:.1f}%</small></div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="status demo"><strong>DEMO DATA · SYNTHETIC DATASET</strong><small>Tracked deterministic fixture. Interface and methodology showcase only; not live public data.</small></div>',unsafe_allow_html=True)

    available=sorted(panel.country_iso3.dropna().astype(str).str.upper().unique())
    default="IND" if "IND" in available else available[0]
    with st.sidebar:
        st.markdown("## Research controls")
        country=st.selectbox("Country",available,index=available.index(st.session_state.get("country",default)) if st.session_state.get("country",default) in available else available.index(default),format_func=lambda x:names.get(x,x),key="country")
        years=sorted(pd.to_numeric(panel.year,errors="coerce").dropna().astype(int).unique())
        latest=int(meta.get("latest_valid_analysis_year") or max(years)) if years else default
        year=st.selectbox("Analysis year",years,index=years.index(st.session_state.get("year",latest)) if st.session_state.get("year",latest) in years else len(years)-1,key="year")
        peer_group=st.selectbox("Comparison set",["Global","Advanced","Emerging"],key="peer_group")
        st.divider()
        if live:
            if st.button("↻ Refresh live data",use_container_width=True): get_live_data.clear(); st.session_state.panel=None; st.session_state.long=None; st.rerun()
            if st.button("Open demo dataset",use_container_width=True): st.session_state.mode="demo"; st.session_state.panel=get_demo_data(); st.session_state.meta=None; st.session_state.long=None; st.rerun()
        else:
            if st.button("Return to live data",type="primary",use_container_width=True): st.session_state.mode="live"; st.session_state.panel=None; st.session_state.long=None; st.session_state.error=None; get_live_data.clear(); st.rerun()
        st.caption(f"Model v{MODEL_VERSION}")
        st.caption("Charts: hover + animation only; zoom/pan disabled.")

    scores,drivers=score_panel(panel)
    row=scores[(scores.country_iso3==country)&(scores.year==year)]
    if row.empty: st.warning("Insufficient data for this country-year."); return
    r=row.iloc[0]; score=float(r.risk_score); band=str(r.risk_band); completeness=float(r.data_completeness)
    previous=scores[(scores.country_iso3==country)&(scores.year==year-1)]
    yoy=score-float(previous.iloc[0].risk_score) if not previous.empty else math.nan
    if peer_group=="Advanced": peer_codes=(yaml.safe_load((CONFIG_PATH/"countries.yaml").read_text(encoding="utf-8")) or {}).get("peer_groups",{}).get("advanced",[])
    elif peer_group=="Emerging": peer_codes=(yaml.safe_load((CONFIG_PATH/"countries.yaml").read_text(encoding="utf-8")) or {}).get("peer_groups",{}).get("emerging",[])
    else: peer_codes=list(scores[scores.year==year].country_iso3.unique())
    peers=scores[(scores.year==year)&(scores.country_iso3.isin(peer_codes))].dropna(subset=["risk_score"])
    position=int((peers.risk_score<score).sum()+1) if not peers.empty else None; median=float(peers.risk_score.median()) if not peers.empty else math.nan

    st.markdown('<div class="section"><h2>Executive view</h2><div class="sub">The decision layer: current position, movement, peers and data confidence.</div></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    with a: st.markdown(f'<div class="card"><div class="eyebrow">Relative risk</div><div class="big">{score:.1f}</div><div class="risk">{band}</div><div class="muted">0–100 model position</div></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="card"><div class="eyebrow">Year-on-year</div><div class="big">{"—" if math.isnan(yoy) else f"{yoy:+.1f}"}</div><div class="muted">Score-point change vs {year-1}</div></div>',unsafe_allow_html=True)
    with c: st.markdown(f'<div class="card"><div class="eyebrow">Peer position</div><div class="big">{f"#{position} / {len(peers)}" if position else "—"}</div><div class="muted">Within selected comparison set</div></div>',unsafe_allow_html=True)
    with d: st.markdown(f'<div class="card"><div class="eyebrow">Score completeness</div><div class="big">{completeness*100:.0f}%</div><div class="muted">Configured score weight observed</div></div>',unsafe_allow_html=True)

    top=top_drivers(drivers,country,year,n=6)
    lead=human(str(top.iloc[0].indicator_code),ind) if not top.empty else "available indicators"
    relation="higher than" if not math.isnan(median) and score>median else "lower than" if not math.isnan(median) else "not comparable to"
    med_text=f"It is {relation} the peer median of {median:.1f}." if not math.isnan(median) else "A peer median is not available for this selection."
    st.markdown(f'<div class="section"><div class="callout"><strong>What does this mean?</strong><br>{names.get(country,country)} sits in the <strong>{band.lower()}</strong> relative-risk band for {year}. The strongest current signal by absolute contribution is <strong>{lead}</strong>. {med_text}<br><br><span class="muted">This score is a relative-positioning research signal. It is not a credit rating, probability of default, forecast or investment recommendation.</span></div></div>',unsafe_allow_html=True)

    # Animated global pulse
    st.markdown('<div class="section"><h2>Global risk pulse</h2><div class="sub">Move through recent years to see the cross-sectional risk landscape. Hover any country for its model score.</div></div>',unsafe_allow_html=True)
    m=scores.dropna(subset=["risk_score"]).copy(); m["country_name"]=m.country_iso3.map(names).fillna(m.country_iso3); recent=sorted(m.year.unique())[-7:]
    scale=[[0,"#3bba9e"],[.35,"#9bd58c"],[.6,"#ffd66b"],[.8,"#ff9f62"],[1,"#ff7180"]]
    frames=[go.Frame(name=str(int(y)),data=[go.Choropleth(locations=m[m.year==y].country_iso3,z=m[m.year==y].risk_score,text=m[m.year==y].country_name,locationmode="ISO-3",colorscale=scale,zmin=0,zmax=100,marker_line_width=0.2,marker_line_color="#27313b",colorbar=dict(title="Risk",len=.72),hovertemplate="%{text}<br>Relative risk: %{z:.1f}<extra></extra>")]) for y in recent]
    last=m[m.year==recent[-1]]
    fig=go.Figure(go.Choropleth(locations=last.country_iso3,z=last.risk_score,text=last.country_name,locationmode="ISO-3",colorscale=scale,zmin=0,zmax=100,marker_line_width=0.2,marker_line_color="#27313b",colorbar=dict(title="Risk",len=.72),hovertemplate="%{text}<br>Relative risk: %{z:.1f}<extra></extra>"),frames=frames)
    fig.update_layout(height=520,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#91a0b2",family="Inter"),geo=dict(scope="world",showframe=False,bgcolor="rgba(0,0,0,0)",landcolor="#121923",showcoastlines=True,coastlinecolor="rgba(145,160,179,.2)",projection_type="natural earth"),dragmode=False,sliders=[dict(currentvalue=dict(prefix="Year · "),steps=[dict(label=str(y),method="animate",args=[[str(y)],{"mode":"immediate","frame":{"duration":500,"redraw":True},"transition":{"duration":250}}]) for y in recent])],updatemenus=[dict(type="buttons",showactive=False,x=.02,y=1.04,buttons=[dict(label="▶ Play",method="animate",args=[None,{"fromcurrent":True,"frame":{"duration":700,"redraw":True},"transition":{"duration":250}}]),dict(label="❚❚ Pause",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False},"transition":{"duration":0}}])])])
    st.plotly_chart(fig,config=chart_config(),use_container_width=True)

    left,right=st.columns([1.05,.95])
    with left:
        st.markdown('<div class="section"><h2>Why is the score here?</h2><div class="sub">Largest model contributions, translated from technical codes into analyst language.</div></div>',unsafe_allow_html=True)
        if top.empty: st.info("No driver decomposition is available for this slice.")
        else:
            for _,x in top.iterrows():
                colour="#ff7180" if float(x.weighted_contribution)>0 else "#55d79b"; raw="—" if pd.isna(x.raw_value) else f"{float(x.raw_value):,.2f} {units(str(x.indicator_code),ind)}".strip()
                st.markdown(f'<div class="driver"><div><b>{human(str(x.indicator_code),ind)}</b><small>{raw} · weight {float(x.weight)*100:.0f}%</small></div><div class="contrib" style="color:{colour}">{float(x.weighted_contribution):+.2f}</div></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section"><h2>Peer context</h2><div class="sub">Relative position is more useful when you can see the comparison set.</div></div>',unsafe_allow_html=True)
        if not peers.empty:
            p=peers.copy(); p["name"]=p.country_iso3.map(names).fillna(p.country_iso3); p=p.sort_values("risk_score").tail(12)
            f=px.bar(p,x="risk_score",y="name",orientation="h",labels={"risk_score":"Relative risk","name":"Country"})
            f.update_traces(marker_color="#8a72d6",hovertemplate="%{y}<br>%{x:.1f}<extra></extra>"); lock_chart(f,360); f.update_xaxes(range=[0,100]); st.plotly_chart(f,config=chart_config(),use_container_width=True)

    st.markdown('<div class="section"><h2>Context over time</h2><div class="sub">Historical relative positioning. Zoom and pan are disabled so the chart remains controlled on mobile.</div></div>',unsafe_allow_html=True)
    hist=scores[scores.country_iso3==country].sort_values("year")
    f=px.line(hist,x="year",y="risk_score",markers=True,labels={"year":"Year","risk_score":"Relative risk"})
    f.update_traces(line=dict(color="#67e8f9",width=2.5),marker=dict(size=5),hovertemplate="Year: %{x}<br>Relative risk: %{y:.1f}<extra></extra>"); lock_chart(f,360); f.update_yaxes(range=[0,100]); st.plotly_chart(f,config=chart_config(),use_container_width=True)

    st.markdown('<div class="section"><h2>Scenario laboratory</h2><div class="sub">One hypothetical shock at a time, interpreted as historical sensitivity rather than a forecast.</div></div>',unsafe_allow_html=True)
    candidates=[r for r in indicator_records() if float(r.get("weight",0))>0 and r.get("code") in panel.columns]
    options=[(r["code"],r.get("label",r["code"]),r.get("unit","") or "units") for r in candidates]
    driver_code,label,unit=st.selectbox("Shock driver",options,format_func=lambda z:f"{z[1]} · {z[2]}",key="scenario_driver")
    amount=st.number_input("Shock amount",value=1.0,step=0.5,help=f"Change in {unit} applied to the historical sensitivity model.",key="scenario_amount")
    if st.button("Run historical sensitivity",type="primary"):
        targets=[r["code"] for r in candidates if r["code"]!=driver_code][:3]
        try: st.session_state.scenario=run_shock_scenario(panel,country,year,driver_code,float(amount),targets)
        except Exception as exc: st.session_state.scenario={"error":str(exc)}
    sc=st.session_state.get("scenario")
    if sc:
        if sc.get("error"): st.error(f"Scenario unavailable: {sc['error']}")
        else:
            delta=float(sc.get("delta",float("nan"))); flag=" · OUT-OF-SAMPLE SHOCK" if sc.get("out_of_sample_shock") else ""
            colour="#ff7180" if delta>0 else "#55d79b"
            st.markdown(f'<div class="card"><div class="eyebrow">Historical sensitivity{flag}</div><div class="big">{float(sc.get("baseline_score",0)):.1f} → {float(sc.get("scenario_score",0)):.1f}</div><div class="risk" style="color:{colour}">{delta:+.1f} pts</div><div class="muted">{sc.get("information_assessment","UNKNOWN")} · {sc.get("model_specification","")}</div></div>',unsafe_allow_html=True)
            for x in sorted(sc.get("indicator_deltas",[]),key=lambda z:abs(float(z.get("estimated_delta",0))),reverse=True)[:3]:
                st.markdown(f'<div class="driver"><div><b>{human(str(x.get("indicator_code")),ind)}</b><small>R² {float(x.get("r_squared",0)):.2f} · n={int(x.get("n_obs",0))}</small></div><div class="contrib">{float(x.get("estimated_delta",0)):+.2f}</div></div>',unsafe_allow_html=True)
            st.caption("Interpretation: pooled historical association; not causal identification, not a forecast.")

    st.markdown('<div class="section"><h2>Evidence & methodology</h2><div class="sub">Inspect the dataset, indicator definitions, model, and outputs.</div></div>',unsafe_allow_html=True)
    t1,t2,t3,t4=st.tabs(["Data quality","Indicator dictionary","Model card","Downloads"])
    with t1:
        q=meta.get("quality",{}) if live else {}
        st.dataframe(pd.DataFrame([{"Check":"Countries","Value":meta.get("country_count",len(available))},{"Check":"Indicators","Value":meta.get("indicator_count",len(ind))},{"Check":"Valid observations","Value":meta.get("valid_observations",q.get("valid_observations",0))},{"Check":"Missing","Value":q.get("missing",0)},{"Check":"Out-of-range","Value":q.get("out_of_range",0)},{"Check":"Coverage","Value":f"{float(q.get('coverage',completeness))*100:.1f}%"}]),use_container_width=True,hide_index=True)
        st.caption(f"Retrieved: {meta.get('retrieved_at','—')} · Latest observation: {meta.get('latest_available_observation','—')} · Latest valid common year: {meta.get('latest_valid_analysis_year',year)}")
    with t2:
        st.dataframe(pd.DataFrame([{"Indicator":r.get("label"),"Code":r.get("code"),"Source":r.get("source"),"Unit":r.get("unit"),"Weight":f"{float(r.get('weight',0))*100:.0f}%","Risk direction":"Higher = higher risk" if float(r.get('risk_direction',1))>0 else "Higher = lower risk"} for r in indicator_records()]),use_container_width=True,hide_index=True)
    with t3:
        st.markdown("**Purpose**\n\nRelative country-risk positioning from public macro indicators.\n\n**Score**\n\nDirection-adjusted weighted cross-sectional z-scores, mapped around a 50-point midpoint and bounded to 0–100. Missing indicators are excluded and remaining configured weights are renormalized.\n\n**Scenario**\n\nPooled-panel bivariate OLS historical sensitivity. It has no structural controls, lags, or causal identification.\n\n**Responsible use**\n\nThis is a research prototype, not a credit rating, default probability, forecast, or investment recommendation.")
    with t4:
        selected=panel[(panel.country_iso3==country)&(panel.year==year)]
        st.download_button("Download country-year data",selected.to_csv(index=False).encode(),file_name=f"{country}_{year}_data.csv",mime="text/csv",use_container_width=True)
        st.download_button("Download all scores",scores.to_csv(index=False).encode(),file_name="country_risk_scores.csv",mime="text/csv",use_container_width=True)
        snap={"country":names.get(country,country),"iso3":country,"year":int(year),"score":score,"band":band,"peer_position":position,"peer_median":median,"mode":st.session_state.mode,"model_version":MODEL_VERSION}
        st.download_button("Download research snapshot",json.dumps(snap,indent=2).encode(),file_name=f"{country}_{year}_snapshot.json",mime="application/json",use_container_width=True)

    st.markdown('<div class="section"><h2>Primary sources</h2><div class="sub">Public sources used by the live collection path.</div></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a: st.markdown('<div class="card"><div class="eyebrow">World Bank</div><div class="risk">Indicators API</div><div class="muted">Annual macroeconomic indicators · https://api.worldbank.org/v2/</div></div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="card"><div class="eyebrow">FRED</div><div class="risk">DFF · US only</div><div class="muted">Effective Federal Funds Rate enrichment · excluded from composite score.</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="footer">Country Risk Intelligence Engine · research / education prototype · model v{MODEL_VERSION}. Live data is fetched from public sources at runtime; demo data is synthetic and explicitly labelled.</div>',unsafe_allow_html=True)

if __name__=="__main__": main()
