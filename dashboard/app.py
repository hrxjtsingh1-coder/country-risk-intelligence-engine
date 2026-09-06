*** Begin Patch
*** Update File: dashboard/app.py
@@
-from src.commentary.generate_commentary import generate_report
-from src.scenario.scenario_engine import run_shock_scenario
-from src.scoring.risk_score import score_panel, top_drivers
+from src.commentary.generate_commentary import generate_report
+from src.scenario.scenario_engine import run_shock_scenario
+from src.scoring.risk_score import score_panel, top_drivers
+
+# Runtime live-data provider
+from src.runtime.live_data import fetch_live_panel
+from src.runtime.provenance import make_provenance
@@
-CONFIG_DIR = ROOT / "config"
-PROCESSED_DIR = ROOT / "data" / "processed"
-PANEL_PATH = PROCESSED_DIR / "panel_wide.csv"
-DEMO_PANEL_PATH = ROOT / "data" / "demo" / "panel_wide.csv"
+CONFIG_DIR = ROOT / "config"
+PROCESSED_DIR = ROOT / "data" / "processed"
+PANEL_PATH = PROCESSED_DIR / "panel_wide.csv"
+DEMO_PANEL_PATH = ROOT / "data" / "demo" / "panel_wide.csv"
+LIVE_CACHE_TTL = 6 * 3600  # 6 hours
@@
-LIVE_METADATA = load_data_metadata(PROCESSED_DIR / "data_metadata.json")
-live_ready = PANEL_PATH.exists() and LIVE_METADATA.get("mode") == "LIVE"
-if "open_demo_dataset" not in st.session_state:
-    st.session_state.open_demo_dataset = False
-if live_ready:
-    DATASET_PATH = PANEL_PATH
-    DATA_STATE = "LIVE"
-elif st.session_state.open_demo_dataset:
-    DATASET_PATH = DEMO_PANEL_PATH
-    DATA_STATE = "DEMO"
-else:
-    DATASET_PATH = None
-    DATA_STATE = "ERROR / UNAVAILABLE"
-USING_DEMO_DATA = DATA_STATE == "DEMO"
-
-if DATASET_PATH is None or not DATASET_PATH.exists():
-    st.markdown(
-        """
-        <div class="card" style="margin-top:24px;">
-            <div class="card-label">LIVE DATA UNAVAILABLE</div>
-            <div class="card-value" style="font-size:24px;">No verified live data vintage is available</div>
-            <div class="card-caption" style="margin-top:10px;">
-                Run <code>python -m src.pipeline.run_all</code> to retrieve public data,
-                or explicitly open the synthetic demo dataset below.
-            </div>
-        </div>
-        """,
-        unsafe_allow_html=True,
-    )
-    if st.button("Open Demo Dataset", type="primary"):
-        st.session_state.open_demo_dataset = True
-        st.rerun()
-    st.stop()
-
-if USING_DEMO_DATA:
-    st.error("DEMO DATA — SYNTHETIC DATASET. Not suitable for economic or investment decisions.")
-else:
-    st.success("LIVE PUBLIC DATA — verified pipeline output loaded.")
-
-panel = load_panel(DATASET_PATH)
+LIVE_METADATA = load_data_metadata(PROCESSED_DIR / "data_metadata.json")
+
+# Session state flags
+if "open_demo_dataset" not in st.session_state:
+    st.session_state.open_demo_dataset = False
+if "live_provenance" not in st.session_state:
+    st.session_state.live_provenance = None
+if "live_panel" not in st.session_state:
+    st.session_state.live_panel = None
+
+
+def _load_live_data_cached(indicators_cfg, countries_cfg):
+    # cached helper to fetch live panel and provenance; TTL approx via Streamlit cache
+    return fetch_live_panel(indicators_cfg, countries_cfg)
+
+
+def try_initialize_live_mode():
+    """Attempt to load live data into session state. Returns DATA_STATE and message."""
+    # If demo explicitly requested, skip
+    if st.session_state.open_demo_dataset:
+        return "DEMO", "Demo dataset requested"
+
+    # If we already have live_panel in session_state, reuse it
+    if st.session_state.live_panel is not None and st.session_state.live_provenance is not None:
+        return "LIVE", "Using cached live data"
+
+    # Otherwise attempt to fetch live data (cached by streamlit caching in fetch_live_panel)
+    try:
+        with st.spinner("Fetching official public data…"):
+            panel, prov = _load_live_data_cached(indicators_cfg, countries_cfg)
+        # basic check
+        if panel is None or panel.empty:
+            return "UNAVAILABLE", "World Bank returned no usable observations"
+
+        st.session_state.live_panel = panel
+        st.session_state.live_provenance = prov
+        return "LIVE", "Live data loaded"
+    except Exception as exc:
+        st.session_state.live_panel = None
+        st.session_state.live_provenance = None
+        st.session_state.live_error = str(exc)
+        return "UNAVAILABLE", str(exc)
+
+
+DATA_STATE = None
+DATASET_PATH = None
+
+if st.session_state.open_demo_dataset:
+    DATA_STATE = "DEMO"
+    DATASET_PATH = DEMO_PANEL_PATH
+else:
+    # try to initialize live
+    state, msg = try_initialize_live_mode()
+    if state == "LIVE":
+        DATA_STATE = "LIVE"
+    else:
+        DATA_STATE = "UNAVAILABLE"
+
+USING_DEMO_DATA = DATA_STATE == "DEMO"
+
+if DATA_STATE == "UNAVAILABLE":
+    st.markdown(
+        """
+        <div class="card" style="margin-top:24px;">
+            <div class="card-label">LIVE DATA UNAVAILABLE</div>
+            <div class="card-value" style="font-size:20px;">Official public data could not be retrieved right now.</div>
+            <div class="card-caption" style="margin-top:10px;">
+                The dashboard attempted to retrieve live World Bank data but the operation failed.
+            </div>
+        </div>
+        """,
+        unsafe_allow_html=True,
+    )
+
+    st.write("Source: World Bank Indicators API v2")
+    if hasattr(st.session_state, "live_error"):
+        with st.expander("Technical details"):
+            st.write(st.session_state.live_error)
+
+    col1, col2 = st.columns(2)
+    with col1:
+        if st.button("Retry"):
+            # clear any previous live cache and retry
+            st.session_state.live_panel = None
+            st.session_state.live_provenance = None
+            st.experimental_rerun()
+    with col2:
+        if st.button("Open Demo Dataset"):
+            st.session_state.open_demo_dataset = True
+            st.experimental_rerun()
+
+    st.stop()
+
+if USING_DEMO_DATA:
+    st.error("DEMO DATA — SYNTHETIC DATASET. Not suitable for economic or investment decisions.")
+    panel = load_panel(DATASET_PATH)
+else:
+    st.success("LIVE PUBLIC DATA — verified runtime fetch loaded.")
+    panel = st.session_state.live_panel
+    # attach LIVE_METADATA from provenance for backward compatibility
+    LIVE_METADATA = st.session_state.live_provenance if isinstance(st.session_state.live_provenance, dict) else (st.session_state.live_provenance.to_dict() if st.session_state.live_provenance is not None else {})
*** End Patch