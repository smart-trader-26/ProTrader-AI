"""
Service layer — pure business logic, no Streamlit imports.

Services compose the primitives in `data/`, `models/`, and `utils/` and expose
typed contracts from `schemas/` to both the Streamlit UI and the future
FastAPI backend. This is the seam Phase 0 carves out so Track A and Track B
land on the same foundation.

Rule: nothing here may `import streamlit`. If a function needs caching,
callers wrap it with `@st.cache_data` / `@st.cache_resource` at the UI layer.
"""
