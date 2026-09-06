"""Streamlit Cloud entrypoint for the Country Risk Intelligence Engine.

BUG FIX (see README "Fixed in this round" for the full writeup):
This file used to do `from dashboard.app import *`. Python caches every
module it imports in `sys.modules`. dashboard/app.py has no `main()` guard —
it's a flat script that renders the page as a side effect of being executed
top to bottom. On the very FIRST load, `import` runs that top-to-bottom
code once and the page renders. Streamlit reruns the *entrypoint file* on
every widget interaction and every new session, but because the module was
already cached, every subsequent `import dashboard.app` was a no-op that
just returned the cached module object WITHOUT re-running any of its
rendering code. Nothing rendered again for that session — which is exactly
"works once, breaks on reopen / a second browser", since Streamlit Cloud
keeps one warm process serving every visitor until it needs to recycle it.

`runpy.run_path` sidesteps the module cache entirely: it re-reads and
re-executes the target file from scratch, every single call, and runs it
with __name__ == "__main__" (matching how `streamlit run dashboard/app.py`
would execute it directly).

Simplest permanent fix, if you're able to touch deployment settings: in
Streamlit Community Cloud's app settings, set "Main file path" to
`dashboard/app.py` directly and delete this wrapper file entirely — then
Streamlit runs that file as the entrypoint itself and this indirection
isn't needed at all. This wrapper exists so the app also works unmodified
if your deployment expects a root-level `streamlit_app.py`.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "dashboard" / "app.py"), run_name="__main__")
