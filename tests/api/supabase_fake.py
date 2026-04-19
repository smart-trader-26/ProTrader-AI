"""
In-memory fake for `supabase.Client` — good enough to exercise the DB-touching
routers without a real Supabase project.

Covers the subset of the PostgREST fluent API the backend uses:
  • `.select(cols, count=...)`                (with `count="exact"`)
  • `.insert(row | rows)`
  • `.upsert(row | rows, on_conflict=..., ignore_duplicates=...)`
  • `.update(patch)`
  • `.delete()`
  • filters: `.eq / .lte / .gte / .is_ / .not_.is_ / .in_`
  • `.order(col, desc=...)`, `.limit(n)`

RLS simulation: when the client is constructed with a `user_id`, every
`select/update/delete` auto-appends a `user_id = <user_id>` filter, and
`insert/upsert` rejects rows whose `user_id` mismatches. For the
`watchlist_tickers` table (no user_id column), the fake joins up through
`watchlists.user_id` so ownership still gates visibility.

Data lives in a shared dict so two FakeSupabase instances bound to
different users see the same underlying store — exactly like real
Supabase with RLS.
"""

from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace
from typing import Any


def _matches(row: dict, filters: list[tuple]) -> bool:
    for op, col, val in filters:
        v = row.get(col)
        if op == "eq":
            if v != val:
                return False
        elif op == "lte":
            if v is None or v > val:
                return False
        elif op == "gte":
            if v is None or v < val:
                return False
        elif op == "is":
            if val == "null" and v is not None:
                return False
            if val is None and v is not None:
                return False
        elif op == "not_is":
            if val == "null" and v is None:
                return False
            if val is None and v is None:
                return False
        elif op == "in":
            if v not in val:
                return False
        else:
            raise AssertionError(f"unknown filter op: {op}")
    return True


class _Query:
    def __init__(self, client: "FakeSupabase", table: str):
        self.client = client
        self.table = table
        self.filters: list[tuple] = []
        self.op: str | None = None
        self.payload: Any = None
        self.count_mode: str | None = None
        self.orderings: list[tuple[str, bool]] = []
        self.limit_n: int | None = None
        self.conflict: str | None = None
        self.ignore_dup: bool = False

    # ─── ops ─────────────────────────────────────────────────────────────
    def select(self, cols: str = "*", count: str | None = None) -> "_Query":
        self.op = "select"
        self.count_mode = count
        return self

    def insert(self, data: Any) -> "_Query":
        self.op = "insert"
        self.payload = data if isinstance(data, list) else [data]
        return self

    def upsert(
        self,
        data: Any,
        on_conflict: str | None = None,
        ignore_duplicates: bool = False,
    ) -> "_Query":
        self.op = "upsert"
        self.payload = data if isinstance(data, list) else [data]
        self.conflict = on_conflict
        self.ignore_dup = ignore_duplicates
        return self

    def update(self, patch: dict) -> "_Query":
        self.op = "update"
        self.payload = patch
        return self

    def delete(self) -> "_Query":
        self.op = "delete"
        return self

    # ─── filters ─────────────────────────────────────────────────────────
    def eq(self, col: str, val: Any) -> "_Query":
        self.filters.append(("eq", col, val))
        return self

    def lte(self, col: str, val: Any) -> "_Query":
        self.filters.append(("lte", col, val))
        return self

    def gte(self, col: str, val: Any) -> "_Query":
        self.filters.append(("gte", col, val))
        return self

    def is_(self, col: str, val: Any) -> "_Query":
        self.filters.append(("is", col, val))
        return self

    @property
    def not_(self) -> "_NotProxy":
        return _NotProxy(self)

    def in_(self, col: str, vals: list) -> "_Query":
        self.filters.append(("in", col, list(vals)))
        return self

    # ─── modifiers ───────────────────────────────────────────────────────
    def order(self, col: str, desc: bool = False) -> "_Query":
        self.orderings.append((col, desc))
        return self

    def limit(self, n: int) -> "_Query":
        self.limit_n = n
        return self

    # ─── execute ─────────────────────────────────────────────────────────
    def execute(self) -> SimpleNamespace:
        store = self.client.store
        rows = store.setdefault(self.table, [])
        uid = self.client.user_id

        # Auto-apply RLS filter for non-admin clients.
        effective_filters = list(self.filters)
        if uid and not self.client.admin:
            if self.table == "watchlist_tickers":
                owned_ids = {
                    w["id"] for w in store.get("watchlists", [])
                    if w.get("user_id") == uid
                }
                effective_filters.append(("in", "watchlist_id", list(owned_ids)))
            elif self.table == "user_profiles":
                effective_filters.append(("eq", "id", uid))
            elif any(c == "user_id" for c in _columns_of(self.table)):
                effective_filters.append(("eq", "user_id", uid))

        if self.op == "select":
            matched = [copy.deepcopy(r) for r in rows if _matches(r, effective_filters)]
            for col, desc in reversed(self.orderings):
                matched.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=desc)
            total = len(matched)
            if self.limit_n is not None:
                matched = matched[: self.limit_n]
            return SimpleNamespace(data=matched, count=total if self.count_mode else None)

        if self.op == "insert":
            inserted = []
            for raw in self.payload:
                row = _normalise_row(self.table, raw, store)
                if uid and not self.client.admin and "user_id" in _columns_of(self.table):
                    if row.get("user_id") != uid:
                        raise PermissionError(f"RLS: can't insert foreign user_id on {self.table}")
                rows.append(row)
                inserted.append(copy.deepcopy(row))
            return SimpleNamespace(data=inserted, count=None)

        if self.op == "upsert":
            keys = [k.strip() for k in (self.conflict or "").split(",") if k.strip()]
            inserted = []
            for raw in self.payload:
                row = _normalise_row(self.table, raw, store)
                if uid and not self.client.admin and "user_id" in _columns_of(self.table):
                    if row.get("user_id") != uid:
                        raise PermissionError(f"RLS: foreign user_id")
                existing_idx = None
                if keys:
                    for i, r in enumerate(rows):
                        if all(r.get(k) == row.get(k) for k in keys):
                            existing_idx = i
                            break
                if existing_idx is not None:
                    if self.ignore_dup:
                        continue
                    rows[existing_idx].update(row)
                    inserted.append(copy.deepcopy(rows[existing_idx]))
                else:
                    rows.append(row)
                    inserted.append(copy.deepcopy(row))
            return SimpleNamespace(data=inserted, count=None)

        if self.op == "update":
            updated = []
            for r in rows:
                if _matches(r, effective_filters):
                    for k, v in self.payload.items():
                        r[k] = v
                    updated.append(copy.deepcopy(r))
            return SimpleNamespace(data=updated, count=None)

        if self.op == "delete":
            keep = []
            removed = []
            for r in rows:
                if _matches(r, effective_filters):
                    removed.append(copy.deepcopy(r))
                else:
                    keep.append(r)
            store[self.table] = keep
            return SimpleNamespace(data=removed, count=None)

        raise AssertionError(f"_Query executed without op: table={self.table}")


class _NotProxy:
    def __init__(self, q: _Query):
        self._q = q

    def is_(self, col: str, val: Any) -> _Query:
        self._q.filters.append(("not_is", col, val))
        return self._q


class FakeSupabase:
    """Pretends to be `supabase.Client` for test purposes."""

    def __init__(self, store: dict, user_id: str | None = None, admin: bool = False):
        self.store = store
        self.user_id = user_id
        self.admin = admin
        # Mimic the `client.postgrest.auth(token)` surface the real client
        # exposes so router code calling `get_user_client()` works unchanged.
        self.postgrest = SimpleNamespace(auth=lambda token: None)

    def table(self, name: str) -> _Query:
        return _Query(self, name)


# ─── helpers ───────────────────────────────────────────────────────────────


_ID_COUNTERS: dict[int, itertools.count] = {}


def _columns_of(table: str) -> set[str]:
    """Cheap schema stand-in — which tables carry a `user_id` column."""
    if table in {"watchlists", "alerts", "predictions", "backtests", "paper_fills", "paper_positions"}:
        return {"user_id"}
    if table == "user_profiles":
        return {"id"}
    return set()


def _normalise_row(table: str, raw: dict, store: dict) -> dict:
    """Fill server-side defaults (id, created_at, added_at) a real Postgres
    would. Auto-increment id comes from a per-store counter so rows stay
    uniquely identifiable across inserts."""
    from datetime import UTC, datetime

    out = dict(raw)
    if "id" not in out and table != "watchlist_tickers" and table != "user_profiles":
        counter = _ID_COUNTERS.setdefault(id(store), itertools.count(1))
        out["id"] = next(counter)
    now = datetime.now(UTC).isoformat()
    if table in {"watchlists", "alerts", "backtests", "paper_fills", "user_profiles"}:
        out.setdefault("created_at", now)
    if table == "watchlist_tickers":
        out.setdefault("added_at", now)
    if table == "predictions":
        out.setdefault("actual_price", None)
        out.setdefault("hit", None)
    if table == "alerts":
        out.setdefault("triggered_at", None)
    return out


def install(monkeypatch) -> dict:
    """Point `db.supabase_client` at a fresh in-memory store. Returns the
    store dict so tests can inspect / seed rows directly.

    Routers capture `get_user_client` / `get_admin_client` at import time
    via `from db.supabase_client import ...`, so we patch BOTH the source
    module and every router's local binding. Adding a new DB-touching
    router means adding it to `_IMPORT_SITES` below.
    """
    store: dict = {}

    def _fake_admin() -> FakeSupabase:
        return FakeSupabase(store, admin=True)

    def _fake_user(access_token: str) -> FakeSupabase:
        import jwt as _jwt

        # Decode without verification — the router already verified it.
        claims = _jwt.decode(access_token, options={"verify_signature": False})
        return FakeSupabase(store, user_id=claims["sub"], admin=False)

    # Canonical module.
    from db import supabase_client as sb

    monkeypatch.setattr(sb, "get_admin_client", _fake_admin)
    monkeypatch.setattr(sb, "get_user_client", _fake_user)
    monkeypatch.setattr(sb, "is_configured", lambda: True)
    monkeypatch.setattr(sb, "has_service_role", lambda: True)

    # Each router's local binding.
    _IMPORT_SITES = [
        "api.routers.watchlists",
        "api.routers.alerts",
        "api.routers.auth",
    ]
    for mod_path in _IMPORT_SITES:
        mod = __import__(mod_path, fromlist=["*"])
        if hasattr(mod, "get_user_client"):
            monkeypatch.setattr(f"{mod_path}.get_user_client", _fake_user)
        if hasattr(mod, "get_admin_client"):
            monkeypatch.setattr(f"{mod_path}.get_admin_client", _fake_admin)
        # The routers import `supabase_client as sb` and call `sb.is_configured()`;
        # patching on sb itself (above) is enough for those.

    return store
