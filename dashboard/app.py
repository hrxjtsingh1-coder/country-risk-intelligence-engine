from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from src.runtime.live_data import fetch_live_data, create_wide_panel_from_long, indicator_records
from src.scoring.risk_score import score_panel, top_drivers
from src.scenario.scenario_engine import run_shock_scenario

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "panel_wide.csv"
CFG = ROOT / "config"

st.set_page_config(page_title="Country Risk Intelligence Engine", page_icon="◎", layout="wide")

@st.cache_data(ttl=21600, show_spinner=False)
def live_bundle() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    long_panel, meta = fetch_live_data(start_year=2012, end_year=pd.Timestamp.utcnow().year)
    return long_panel, meta, create_wide_panel_from_long(long_panel)

@st.cache_data(ttl=3600, show_spinner=False)
def demo_panel() -> pd.DataFrame:
    return pd.read_csv(DEMO)

def css() -> None:
    st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 0 0,rgba(103,232,249,.08),transparent 26%),radial-gradient(circle at 100% 0,rgba(167,139,250,.09),transparent 30%),#05070b;color:#f4f7fb}.block-container{max-width:1500px;padding-top:1rem}.hero{border:1px solid rgba(148,163,184,.15);border-radius:24px;padding:28px;background:linear-gradient(135deg,rgba(17,25,35,.96),rgba(8,12,18,.95));box-shadow:0 24px 80px rgba(0,0,0,.28);animation:rise .5s ease}.k{font:600 10px ui-monospace,monospace;letter-spacing:.18em;color:#67e8f9}.hero h1{margin:8px 0;font-size:clamp(32px,5vw,60px);letter-spacing:-.05em}.hero h1 span{background:linear-gradient(95deg,#fff,#c8f8fa,#bba8ff);-webkit-background-clip:text;color:transparent}.hero p{max-width:920px;color:#91a0b2;line-height:1.65}.status{margin:13px 0;padding:12px 15px;border:1px solid rgba(148,163,184,.15);border-radius:14px;background:rgba(12,17,24,.86)}.status strong{font-size:11px}.status small{display:block;color:#91a0b2;margin-top:4px}.live{border-color:rgba(86,214,160,.3)}.live strong{color:#56d6a0}.demo{border-color:rgba(255,215,107,.3)}.demo strong{color:#ffd76b}.bad{border-color:rgba(255,113,128,.3)}.bad strong{color:#ff7180}.section{margin-top:24px}.section h2{margin:0;font-size:20px}.sub{color:#91a0b2;font-size:12px;margin:4px 0 12px}.card{border:1px solid rgba(148,163,184,.15);border-radius:17px;padding:16px;background:linear-gradient(145deg,rgba(17,25,35,.92),rgba(10,14,20,.88));height:100%}.ey{font:600 10px ui-monospace,monospace;color:#91a0b2;letter-spacing:.1em}.big{font-size:35px;font-weight:800;margin-top:6px;letter-spacing:-.04em}.risk{font-size:20px;font-weight:800}.driver{display:flex;justify-content:space-between;gap:12px;padding:10px 0;border-bottom:1px solid rgba(148,163,184,.08)}.driver:last-child{border:0}.driver b{font-size:13px}.driver small{display:block;color:#91a0b2;margin-top:3px}.contrib{font:650 13px ui-monospace,monospace}.callout{border:1px solid rgba(103,232,249,.16);border-radius:16px;padding:15px;background:rgba(103,232,249,.04);color:#dbe6ef;line-height:1.6}.footer{border-top:1px solid rgba(148,163,184,.12);margin-top:28px;padding-top:14px;color:#667386;font-size:10px}@keyframes rise{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}@media(max-width:700px){.hero{padding:20px 16px;border-radius:18px}.hero h1{font-size:31px}.big{font-size:29px}.block-container{padding-left:.7rem;padding-right:.7rem}}
</style>
""", unsafe_allow_html=True)

def cfg() -> tuple[dict[str,str], dict[str,dict[str,Any]], dict[str,list[str]]]:
    c=yaml.safe_load((CFG/"countries.yaml").read_text(encoding="utf-8")) or {}
    names={str(x["iso3"]):str(x["name"]) for x in c.get("countries",[]) if x.get("iso3")}
    peers={str(k):[str(v) for v in val] for k,val in c.get("peer_groups",{}).items()}
    inds={str(x["code"]):x for x in indicator_records()}
    return names,inds,peers

def chart_cfg() -> dict[str,Any]:
    return {"displayModeBar":False,"responsive":True,"scrollZoom":False,"doubleClick":False}

def main() -> None:
    css(); names,inds,peer_groups=cfg()
    st.session_state.setdefault("mode","live"); st.session_state.setdefault("panel",None); st.session_state.setdefault("meta",None); st.session_state.setdefault("long",None); st.session_state.setdefault("error",None); st.session_state.setdefault("scenario",None)
    if st.session_state.mode=="live" and st.session_state.panel is None:
        try:
            long,meta,panel=live_bundle(); st.session_state.long=long; st.session_state.meta=meta; st.session_state.panel=panel; st.session_state.error=None
        except Exception as exc: st.session_state.error=str(exc)
    if st.session_state.mode=="demo" and st.session_state.panel is None: st.session_state.panel=demo_panel()
    st.markdown('<div class="hero"><div class="k">GLOBAL MACRO · COUNTRY RISK INTELLIGENCE</div><h1>Country Risk <span>Intelligence Engine</span></h1><p>Explore relative country-risk positioning, understand what drives it, compare peers, watch the global landscape evolve, and run transparent historical-sensitivity scenarios.</p></div>',unsafe_allow_html=True)
    panel=st.session_state.panel
    if panel is None or panel.empty:
        st.markdown('<div class="status bad"><strong>● LIVE DATA UNAVAILABLE</strong><small>Official public data could not be verified. No synthetic data is substituted silently.</small></div>',unsafe_allow_html=True)
        with st.expander("Technical details"): st.code(st.session_state.error or "Unknown error")
        a,b=st.columns(2)
        with a:
            if st.button("↻ Retry live data",type="primary",use_container_width=True): live_bundle.clear(); st.session_state.panel=None; st.session_state.meta=None; st.session_state.long=None; st.session_state.error=None; st.rerun()
        with b:
            if st.button("Open Demo Dataset",use_container_width=True): st.session_state.mode="demo"; st.session_state.panel=demo_panel(); st.rerun()
        return
    live=st.session_state.mode=="live"; meta=st.session_state.meta or {}
    if live:
        q=meta.get("quality",{}); latest=meta.get("latest_valid_analysis_year") or meta.get("latest_available_observation") or "—"; when=meta.get("retrieved_at","—")
        st.markdown(f'<div class="status live"><strong>● LIVE PUBLIC DATA</strong><small>World Bank Indicators API + FRED US-only enrichment · Retrieved {when} · Latest valid year {latest} · valid observation coverage {float(q.get("coverage",0))*100:.1f}%</small></div>',unsafe_allow_html=True)
    else: st.markdown('<div class="status demo"><strong>● DEMO DATA · SYNTHETIC DATASET</strong><small>Tracked fixture for interface/methodology demonstration only.</small></div>',unsafe_allow_html=True)
    countries=sorted(panel.country_iso3.dropna().astype(str).str.upper().unique()); years=sorted(pd.to_numeric(panel.year,errors="coerce").dropna().astype(int).unique())
    with st.sidebar:
        st.markdown("## Research controls")
        country=st.selectbox("Country",countries,index=countries.index("IND") if "IND" in countries else 0,format_func=lambda x:names.get(x,x))
        year=st.selectbox("Analysis year",years,index=len(years)-1)
        peer=st.selectbox("Comparison set",["Global","Advanced","Emerging"])
        st.divider()
        if live:
            if st.button("↻ Refresh live data",use_container_width=True): live_bundle.clear(); st.session_state.panel=None; st.session_state.meta=None; st.session_state.long=None; st.rerun()
            if st.button("Open demo dataset",use_container_width=True): st.session_state.mode="demo"; st.session_state.panel=demo_panel(); st.session_state.meta=None; st.session_state.long=None; st.rerun()
        else:
            if st.button("Return to live data",type="primary",use_container_width=True): st.session_state.mode="live"; st.session_state.panel=None; st.session_state.meta=None; st.session_state.long=None; st.session_state.error=None; live_bundle.clear(); st.rerun()
    scores,drivers=score_panel(panel); current=scores[(scores.country_iso3==country)&(scores.year==year)]
    if current.empty: st.warning("Insufficient data for this country-year."); return
    r=current.iloc[0]; score=float(r.risk_score); band=str(r.risk_band); comp=float(r.data_completeness); prev=scores[(scores.country_iso3==country)&(scores.year==year-1)]; yoy=score-float(prev.iloc[0].risk_score) if not prev.empty and pd.notna(prev.iloc[0].risk_score) else math.nan
    peer_codes=peer_groups.get("advanced",[]) if peer=="Advanced" else peer_groups.get("emerging",[]) if peer=="Emerging" else countries; peers=scores[(scores.year==year)&(scores.country_iso3.isin(peer_codes))].dropna(subset=["risk_score"]); med=float(peers.risk_score.median()) if not peers.empty else math.nan; pos=int((peers.risk_score<score).sum()+1) if not peers.empty else None; top=top_drivers(drivers,country,year,6)
    st.markdown('<div class="section"><h2>Executive view</h2><div class="sub">Current position, movement, peers and evidence quality.</div></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    with a: st.markdown(f'<div class="card"><div class="ey">RELATIVE RISK</div><div class="big">{score:.1f}</div><div class="risk">{band}</div><div class="muted">model position · 0–100</div></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="card"><div class="ey">YEAR-ON-YEAR</div><div class="big">{"—" if math.isnan(yoy) else f"{yoy:+.1f}"}</div><div class="muted">score-point change vs {year-1}</div></div>',unsafe_allow_html=True)
    with c: st.markdown(f'<div class="card"><div class="ey">PEER POSITION</div><div class="big">{f"#{pos} / {len(peers)}" if pos else "—"}</div><div class="muted">selected comparison set</div></div>',unsafe_allow_html=True)
    with d: st.markdown(f'<div class="card"><div class="ey">EVIDENCE COVERAGE</div><div class="big">{comp*100:.0f}%</div><div class="muted">configured weight observed</div></div>',unsafe_allow_html=True)
    lead=str(top.iloc[0]["label"]) if not top.empty else "available indicators"; peer_text=f"Peer median: {med:.1f}." if not math.isnan(med) else "Peer median unavailable."
    st.markdown(f'<div class="section"><div class="callout"><strong>{names.get(country,country)} · {year}</strong><br>Largest current absolute model contribution: <strong>{lead}</strong>. {peer_text}<br><span class="muted">Relative-risk positioning is a research signal, not a credit rating, forecast or investment recommendation.</span></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section"><h2>Global risk pulse</h2><div class="sub">Animated year-by-year world view of relative risk.</div></div>',unsafe_allow_html=True)
    md=scores.dropna(subset=["risk_score"]).copy(); md["name"]=md.country_iso3.map(names).fillna(md.country_iso3)
    if md.year.nunique()>1:
        f=px.choropleth(md,locations="country_iso3",color="risk_score",hover_name="name",animation_frame="year",locationmode="ISO-3",range_color=(0,100),color_continuous_scale=[[0,"#39b79b"],[.35,"#a4d98b"],[.58,"#ffd66b"],[.78,"#ff9d61"],[1,"#ff7080"]]); f.update_layout(height=520,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",geo=dict(scope="world",showframe=False,bgcolor="rgba(0,0,0,0)",landcolor="#121923")); st.plotly_chart(f,config=chart_cfg(),use_container_width=True)
    st.markdown('<div class="section"><h2>Why is the score here?</h2><div class="sub">Largest direction-adjusted model contributions.</div></div>',unsafe_allow_html=True)
    for _,x in top.iterrows():
        cc="#ff7180" if float(x.weighted_contribution)>0 else "#56d6a0"; unit=inds.get(str(x.indicator_code),{}).get("unit",""); raw="—" if pd.isna(x.raw_value) else f"{float(x.raw_value):,.2f} {unit}".strip()
        st.markdown(f'<div class="driver"><div><b>{x["label"]}</b><small>{raw} · weight {float(x.weight)*100:.0f}% · {x["indicator_code"]}</small></div><div class="contrib" style="color:{cc}">{float(x.weighted_contribution):+.2f}</div></div>',unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        st.markdown('<div class="section"><h2>Peer context</h2><div class="sub">Relative position inside the chosen set.</div></div>',unsafe_allow_html=True)
        p=peers.copy(); p["name"]=p.country_iso3.map(names).fillna(p.country_iso3); f=px.bar(p.sort_values("risk_score").tail(15),x="risk_score",y="name",orientation="h"); f.update_traces(marker_color="#9076df",hovertemplate="%{y}<br>%{x:.1f}<extra></extra>"); f.update_layout(xaxis_range=[0,100],yaxis_title="",xaxis_title="Relative risk"); st.plotly_chart(f,config=chart_cfg(),use_container_width=True)
    with right:
        st.markdown('<div class="section"><h2>Country trajectory</h2><div class="sub">Historical relative-risk positioning.</div></div>',unsafe_allow_html=True)
        h=scores[scores.country_iso3==country].sort_values("year"); f=px.line(h,x="year",y="risk_score",markers=True); f.update_traces(line=dict(color="#67e8f9",width=2.7),hovertemplate="Year %{x}<br>%{y:.1f}<extra></extra>"); f.update_layout(yaxis_range=[0,100]); st.plotly_chart(f,config=chart_cfg(),use_container_width=True)
    st.markdown('<div class="section"><h2>Scenario laboratory</h2><div class="sub">Historical sensitivity, not a forecast.</div></div>',unsafe_allow_html=True)
    opts=[(r["code"],r.get("label",r["code"]),r.get("unit","") or "units") for r in indicator_records() if float(r.get("weight",0) or 0)>0 and r.get("code") in panel.columns]
    if opts:
        drv=st.selectbox("Shock driver",opts,format_func=lambda z:f"{z[1]} · {z[2]}"); amt=st.number_input("Shock amount",value=1.0,step=.5)
        if st.button("Run historical sensitivity",type="primary"):
            try: st.session_state.scenario=run_shock_scenario(panel,country,year,drv[0],float(amt),[z[0] for z in opts if z[0]!=drv[0]][:3])
            except Exception as exc: st.session_state.scenario={"error":str(exc)}
        sc=st.session_state.scenario
        if sc:
            if sc.get("error"): st.error(sc["error"])
            else:
                delta=float(sc["delta"]); col="#ff7180" if delta>0 else "#56d6a0" if delta<0 else "#91a0b2"; st.markdown(f'<div class="card"><div class="ey">HISTORICAL SENSITIVITY</div><div class="big">{float(sc["baseline_score"]):.1f} → {float(sc["scenario_score"]):.1f}</div><div class="risk" style="color:{col}">{delta:+.1f} pts</div><div class="muted">{sc.get("information_assessment","UNKNOWN")} · {sc.get("model_specification","")}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section"><h2>Evidence & methodology</h2><div class="sub">Inspect what the engine used and export results.</div></div>',unsafe_allow_html=True)
    t1,t2,t3,t4=st.tabs(["Data quality","Indicators","Model card","Exports"])
    with t1:
        q=meta.get("quality",{}) if live else {}; st.dataframe(pd.DataFrame([{ "Check":"Countries","Value":int(q.get("countries",len(countries)))},{"Check":"Indicators","Value":int(q.get("indicators",len(indicators)))},{"Check":"Valid observations","Value":int(q.get("valid_observations",0))},{"Check":"Missing","Value":int(q.get("missing",0))},{"Check":"Out-of-range","Value":int(q.get("out_of_range",0))},{"Check":"Coverage","Value":f"{float(q.get('coverage',comp))*100:.1f}%"}]),use_container_width=True,hide_index=True)
    with t2:
        st.dataframe(pd.DataFrame([{ "Indicator":x.get("label"),"Code":x.get("code"),"Source":x.get("source"),"Unit":x.get("unit"),"Weight":f"{float(x.get('weight',0))*100:.0f}%","Direction":"Higher = higher risk" if float(x.get("risk_direction",1))>0 else "Higher = lower risk"} for x in indicator_records()]),use_container_width=True,hide_index=True)
    with t3:
        st.markdown("**Purpose**  \nRelative country-risk positioning from public macro indicators.\n\n**Score**  \nDirection-adjusted weighted cross-sectional z-scores around a 50 midpoint, bounded to 0–100; missing indicators are excluded and remaining weights are renormalized.\n\n**Scenario**  \nPooled-panel bivariate OLS sensitivity analysis; not causal identification or a forecast.")
    with t4:
        sel=panel[(panel.country_iso3==country)&(panel.year==year)]; st.download_button("Download country-year data",sel.to_csv(index=False).encode(),file_name=f"{country}_{year}.csv",mime="text/csv",use_container_width=True); st.download_button("Download all scores",scores.to_csv(index=False).encode(),file_name="risk_scores.csv",mime="text/csv",use_container_width=True)
        report=f"# {names.get(country,country)} — {year}\n\nRisk score: {score:.1f} ({band})\n\nLargest drivers:\n"+"\n".join(f"- {x['label']}: {float(x['weighted_contribution']):+.2f}" for _,x in top.iterrows())
        st.download_button("Download analyst brief",report.encode(),file_name=f"{country}_{year}_brief.md",mime="text/markdown",use_container_width=True)
    st.markdown(f'<div class="footer">Country Risk Intelligence Engine · model 2.0.0 · public-data research prototype · live and synthetic modes are explicitly separated.</div>',unsafe_allow_html=True)

if __name__ == "__main__": main()
