"""Canonical Streamlit entrypoint.

The public application lives in dashboard.public_app so both
`streamlit run dashboard/app.py` and the Streamlit Cloud entrypoint use
exactly the same code path.
"""
from dashboard.public_app import main

main()
