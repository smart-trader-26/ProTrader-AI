"""
FastAPI backend (Track B1).

Carved out so the same `services/` layer that powers the Streamlit UI also
powers the future Next.js frontend. Nothing here imports streamlit; nothing
in `services/` imports `api`. The two are siblings, not parent/child.

App factory: `from api.main import create_app`.
"""
