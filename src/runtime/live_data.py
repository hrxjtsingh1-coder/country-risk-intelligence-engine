"""Runtime public-data collection for the Country Risk Intelligence Engine."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
INDICATORS_PATH = ROOT / "config" / "indicators.yaml"
COUNTRIES_PATH = ROOT / "config" / "countries.yaml"
WB_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
LOG = logging.getLogger("country-risk.live")
TIMEOUT = 30


def indicator_records() -> list[dict[str, Any]]:
    cfg = yaml.safe_load(INDICATORS_PATH.read_text(encoding="utf-8")) or {}
    return [x for x in cfg.get("indicators", []) if isinstance(x, dict) and x.get("code")]


def country_records() -> list[dict[str, Any]]:
    cfg = yaml.safe_load(COUNTRIES_PATH.read_text(encoding="utf-8")) or {}
    return [x for x in cfg.get("countries", []) if isinstance(x, dict) and x.get("iso3")]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent":"country-risk-intelligence-engine/1.3 (+https://github.com/hrxjtsingh1-coder/country-risk-intelligence-engine)","Accept":"application/json,text/csv;q=0.9,*/*;q=0.8"})
    retry = Retry(total=3, connect=3, read=3, backoff_factor=.7, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset({"GET"}))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _wb_indicator(session: requests.Session, code: str, start: int, end: int) -> pd.DataFrame:
    """Fetch one indicator for all actual countries, excluding WB aggregates."""
    r = session.get(WB_URL.format(indicator=code), params={"format":"json","per_page":40000,"date":f"{start}:{end}"}, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    cols=["country_iso3","country_name","indicator_code","year","value","source"]
    if not isinstance(payload,list) or len(payload)<2 or not payload[1]:
        return pd.DataFrame(columns=cols)
    rows=[]
    for rec in payload[1]:
        iso=str(rec.get("countryiso3code") or "").upper()
        region=((rec.get("region") or {}).get("id") or "").upper()
        if len(iso)!=3 or region=="NA":
            continue
        try: year=int(rec.get("date"))
        except (TypeError,ValueError): continue
        rows.append({"country_iso3":iso,"country_name":str((rec.get("country") or {}).get("value") or iso),"indicator_code":code,"year":year,"value":rec.get("value"),"source":"World Bank Indicators API"})
    return pd.DataFrame(rows,columns=cols)


def _fred_dff(session: requests.Session, start: int, end: int) -> pd.DataFrame:
    try:
        r=session.get(FRED_URL.format(series="DFF"),timeout=TIMEOUT); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text));
        if df.empty:return pd.DataFrame()
        d="observation_date" if "observation_date" in df.columns else df.columns[0]; v="DFF" if "DFF" in df.columns else df.columns[1]
        df[d]=pd.to_datetime(df[d],errors="coerce"); df[v]=pd.to_numeric(df[v],errors="coerce"); df=df.dropna(subset=[d,v]); df["year"]=df[d].dt.year.astype(int); df=df[df.year.between(start-1,end)]
        a=df.groupby("year",as_index=False)[v].mean().rename(columns={v:"value"}); a["value"]=a.value.diff()*100.; a=a[a.year.between(start,end)].copy(); a["country_iso3"]="USA"; a["country_name"]="United States"; a["indicator_code"]="POLICY_RATE_YOY_CHANGE_BPS"; a["source"]="FRED DFF; annual-average change in basis points"
        return a[["country_iso3","country_name","indicator_code","year","value","source"]]
    except Exception as exc:
        LOG.warning("FRED DFF unavailable: %s",exc); return pd.DataFrame()


def _fx_from_raw(raw: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    if raw.empty:return pd.DataFrame()
    x=raw.copy(); x["value"]=pd.to_numeric(x.value,errors="coerce"); x=x.dropna(subset=["country_iso3","year","value"]).sort_values(["country_iso3","year"]); x["value"]=x.groupby("country_iso3").value.pct_change()*100.; x["indicator_code"]="FX_YOY_DEPRECIATION_PCT"; x["source"]="World Bank; derived from PA.NUS.FCRF"; return x[x.year.between(start,end)][["country_iso3","country_name","indicator_code","year","value","source"]]


def _clean_long(df: pd.DataFrame) -> pd.DataFrame:
    cols=["country_iso3","country_name","indicator_code","year","value","source","flag"]
    if df.empty:return pd.DataFrame(columns=cols)
    bounds={x["code"]:(x.get("min"),x.get("max")) for x in indicator_records()}; out=df.copy(); out["country_iso3"]=out.country_iso3.astype(str).str.upper().str.strip(); out["country_name"]=out.get("country_name","").fillna("").astype(str); out["indicator_code"]=out.indicator_code.astype(str).str.strip(); out["year"]=pd.to_numeric(out.year,errors="coerce"); out["value"]=pd.to_numeric(out.value,errors="coerce")
    flags=[]
    for _,row in out.iterrows():
        if pd.isna(row.value): flags.append("missing"); continue
        lo,hi=bounds.get(row.indicator_code,(None,None)); flags.append("out_of_range" if (lo is not None and row.value<float(lo)) or (hi is not None and row.value>float(hi)) else "ok")
    out["flag"]=flags; out=out.dropna(subset=["country_iso3","indicator_code","year"]); out["year"]=out.year.astype(int)
    return out[cols].sort_values(["country_iso3","indicator_code","year"]).drop_duplicates(["country_iso3","indicator_code","year"],keep="last").reset_index(drop=True)


def latest_valid_analysis_year(long: pd.DataFrame, indicators: list[str], min_coverage: float=.55, min_countries:int=30) -> int|None:
    ok=long[long.flag=="ok"] if not long.empty else pd.DataFrame()
    if ok.empty:return None
    for y in sorted(ok.year.unique(),reverse=True):
        p=ok[ok.year==y]; countries=p.country_iso3.nunique(); share=p.indicator_code.isin(indicators).mean() if not p.empty else 0
        if countries>=min_countries and share>=min_coverage:return int(y)
    return int(max(ok.year.unique()))


def fetch_live_data(start_year:int=2012, end_year:int|None=None)->tuple[pd.DataFrame,dict[str,Any]]:
    end_year=end_year or pd.Timestamp.utcnow().year; s=_session(); cfg=indicator_records(); frames=[]; statuses=[]
    for item in cfg:
        code=str(item["code"]); source="FRED" if item.get("source")=="fred_us_only" else "World Bank"
        try:
            if code=="FX_YOY_DEPRECIATION_PCT": frame=_fx_from_raw(_wb_indicator(s,"PA.NUS.FCRF",start_year-1,end_year),start_year,end_year)
            elif item.get("source")=="fred_us_only": frame=_fred_dff(s,start_year,end_year)
            elif item.get("world_bank"): frame=_wb_indicator(s,str(item["world_bank"]),start_year,end_year); frame["indicator_code"]=code if not frame.empty else frame.get("indicator_code")
            else: frame=pd.DataFrame()
            statuses.append({"indicator":code,"source":source,"status":"PASS" if not frame.empty else "NO_DATA","observations":int(len(frame))})
            if not frame.empty: frames.append(frame)
        except Exception as exc:
            LOG.warning("%s failed: %s",code,exc); statuses.append({"indicator":code,"source":source,"status":"ERROR","observations":0})
    long=_clean_long(pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()); scoring=[x["code"] for x in cfg if float(x.get("weight",0))>0]; latest=latest_valid_analysis_year(long,scoring)
    q={"countries":int(long.country_iso3.nunique()) if not long.empty else 0,"indicators":int(long.indicator_code.nunique()) if not long.empty else 0,"observations":int(len(long)),"valid_observations":int((long.flag=="ok").sum()) if not long.empty else 0,"missing":int((long.flag=="missing").sum()) if not long.empty else 0,"out_of_range":int((long.flag=="out_of_range").sum()) if not long.empty else 0,"coverage":float((long.flag=="ok").mean()) if not long.empty else 0.0}
    meta={"run_id":pd.Timestamp.utcnow().strftime("live-%Y%m%dT%H%M%SZ"),"mode":"LIVE","retrieved_at":pd.Timestamp.utcnow().isoformat(),"requested_period":f"{start_year}–{end_year}","latest_available_observation":int(long.year.max()) if not long.empty else None,"latest_valid_analysis_year":latest,"country_count":q["countries"],"indicator_count":q["indicators"],"observations_received":q["observations"],"valid_observations":q["valid_observations"],"quality":q,"source_status":statuses,"sources":["World Bank Indicators API","FRED DFF (US-only enrichment)"]}
    if long.empty: raise RuntimeError("Official public sources returned no usable observations")
    return long,meta


def create_wide_panel_from_long(long:pd.DataFrame)->pd.DataFrame:
    if long.empty:return pd.DataFrame(columns=["country_iso3","year"])
    x=long.copy(); x["value_valid"]=x.value.where(x.flag=="ok"); wide=x.pivot_table(index=["country_iso3","year"],columns="indicator_code",values="value_valid",aggfunc="last").reset_index(); wide.columns.name=None; return wide.sort_values(["country_iso3","year"]).reset_index(drop=True)


def data_quality(long:pd.DataFrame,analysis_year:int|None=None)->dict[str,Any]:
    x=long if analysis_year is None else long[long.year==analysis_year]
    return {"countries":int(x.country_iso3.nunique()) if not x.empty else 0,"indicators":int(x.indicator_code.nunique()) if not x.empty else 0,"observations":int(len(x)),"missing":int((x.flag=="missing").sum()) if not x.empty else 0,"out_of_range":int((x.flag=="out_of_range").sum()) if not x.empty else 0,"coverage":float((x.flag=="ok").mean()) if not x.empty else 0.0}
