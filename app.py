from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from authlib.integrations.starlette_client import OAuth, OAuthError
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware
try:
    from starlette.middleware.proxy_headers import ProxyHeadersMiddleware
except ModuleNotFoundError:  # pragma: no cover - fallback for older Starlette
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "portfolio.db"

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
load_dotenv(BASE_DIR / ".env")
env_file = BASE_DIR / f".env.{APP_ENV}"
if env_file.exists():
    load_dotenv(env_file, override=True)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip() or os.getenv("SECRET_KEY", "").strip()
SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "").strip().lower() in ("1", "true", "yes")
PROXY_HEADERS = os.getenv("PROXY_HEADERS", "").strip().lower() in ("1", "true", "yes")

if not SESSION_SECRET:
    # Fallback for local-only use; override in production via env var.
    SESSION_SECRET = "dev-session-secret"

app = FastAPI()
if PROXY_HEADERS:
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=SESSION_HTTPS_ONLY if APP_ENV == "production" else False,
)

oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

NAME_OVERRIDES = {
    "TSLY": "YieldMax TSLA Option Income Strategy ETF",
    "SBIT": "ProShares UltraShort Bitcoin ETF",
    "ETHD": "ProShares UltraShort Ether ETF",
}

NAVER_URL = "https://finance.naver.com/item/sise.naver?code={ticker}"
YAHOO_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
YAHOO_HTML_URL = "https://finance.yahoo.com/quote/{ticker}"
STOOQ_URL = "https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"
INVESTING_SEARCH_URL = "https://www.investing.com/search/service/searchTopBar"
INVESTING_BASE_URL = "https://www.investing.com"


def parse_number(text: str) -> float:
    cleaned = re.sub(r"[^0-9.-]", "", text)
    if cleaned in ("", ".", "-", "-."):
        raise ValueError("invalid number")
    return float(cleaned)


def format_ticker(raw: str) -> str:
    return raw.strip().upper()


def is_korean_ticker(ticker: str) -> bool:
    if re.fullmatch(r"\d{6}", ticker):
        return True
    # Some Korean ETFs use mixed codes like 0091P0 or 0023A0
    return len(ticker) == 6 and ticker[0].isdigit()


def is_cash_ticker(ticker: str) -> bool:
    return ticker.upper() == "NA"


def fetch_naver_quote(ticker: str) -> dict:
    url = NAVER_URL.format(ticker=ticker)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }
    with httpx.Client(timeout=10.0, headers=headers) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="naver fetch failed")

    soup = BeautifulSoup(resp.text, "html.parser")
    name_el = soup.select_one("div.wrap_company h2 a") or soup.select_one("div.h_company h2 a")
    price_el = soup.select_one("#_nowVal")

    if not name_el or not price_el:
        raise HTTPException(status_code=502, detail="naver parse failed")

    name = name_el.get_text(strip=True)
    price_text = price_el.get_text(strip=True)
    try:
        price = parse_number(price_text)
    except ValueError:
        raise HTTPException(status_code=502, detail="naver price invalid")

    return {
        "provider": "naver",
        "ticker": ticker,
        "name": name,
        "price": price,
        "currency": "KRW",
    }


def fetch_yahoo_quote(ticker: str) -> dict:
    url = YAHOO_URL.format(ticker=ticker)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json,text/plain,*/*",
    }
    with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="yahoo fetch failed")

    try:
        payload = resp.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="yahoo json invalid")

    result = payload.get("quoteResponse", {}).get("result", [])
    if not result:
        raise HTTPException(status_code=404, detail="yahoo symbol not found")

    quote = result[0]
    price = quote.get("regularMarketPrice")
    name = quote.get("shortName") or quote.get("longName") or ticker
    currency = quote.get("currency") or ""

    if price is None:
        raise HTTPException(status_code=502, detail="yahoo price missing")

    return {
        "provider": "yahoo",
        "ticker": ticker,
        "name": name,
        "price": float(price),
        "currency": currency,
    }


def fetch_yahoo_quote_html(ticker: str) -> dict:
    url = YAHOO_HTML_URL.format(ticker=ticker)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
    }
    with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="yahoo html fetch failed")

    soup = BeautifulSoup(resp.text, "html.parser")
    price_el = soup.select_one(
        f"fin-streamer[data-field='regularMarketPrice'][data-symbol='{ticker}']"
    )
    if not price_el:
        price_el = soup.select_one("fin-streamer[data-field='regularMarketPrice']")
    name_el = soup.select_one("h1")
    currency_el = soup.select_one("fin-streamer[data-field='currency']")

    if not price_el:
        raise HTTPException(status_code=502, detail="yahoo html parse failed")

    price_text = price_el.get_text(strip=True)
    try:
        price = parse_number(price_text)
    except ValueError:
        raise HTTPException(status_code=502, detail="yahoo html price invalid")

    symbol_attr = price_el.get("data-symbol")
    if symbol_attr and symbol_attr.upper() != ticker.upper():
        raise HTTPException(status_code=502, detail="yahoo html symbol mismatch")

    name = name_el.get_text(strip=True) if name_el else ticker
    currency = currency_el.get_text(strip=True) if currency_el else "USD"

    return {
        "provider": "yahoo_html",
        "ticker": ticker,
        "name": name,
        "price": float(price),
        "currency": currency,
    }


def fetch_stooq_quote(ticker: str) -> dict:
    symbol = f"{ticker.lower()}.us"
    url = STOOQ_URL.format(symbol=symbol)
    headers = {"User-Agent": "Mozilla/5.0"}
    with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="stooq fetch failed")

    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        raise HTTPException(status_code=502, detail="stooq csv empty")

    header = lines[0].split(",")
    data = lines[1].split(",")
    if len(data) != len(header):
        raise HTTPException(status_code=502, detail="stooq csv invalid")

    record = dict(zip(header, data))
    close = record.get("Close")
    if not close or close == "N/A":
        raise HTTPException(status_code=404, detail="stooq symbol not found")

    try:
        price = float(close)
    except ValueError:
        raise HTTPException(status_code=502, detail="stooq price invalid")

    name = ticker
    try:
        quote = fetch_investing_search_quote(ticker)
        name = quote.get("name") or quote.get("symbol") or ticker
    except HTTPException:
        pass

    return {
        "provider": "stooq",
        "ticker": ticker,
        "name": name,
        "price": price,
        "currency": "USD",
    }


def fetch_investing_quote(ticker: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json,text/plain,*/*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.investing.com/",
    }
    quote = fetch_investing_search_quote(ticker, headers)
    link = quote.get("link") or quote.get("tag") or ""
    if not link:
        raise HTTPException(status_code=502, detail="investing link missing")

    url = f"{INVESTING_BASE_URL}{link}"
    with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
        page = client.get(url)
    if page.status_code != 200:
        raise HTTPException(status_code=502, detail="investing page failed")

    soup = BeautifulSoup(page.text, "html.parser")
    price_el = soup.find(attrs={"data-test": "instrument-price-last"})
    if not price_el:
        price_el = soup.select_one("#last_last")
    if not price_el:
        raise HTTPException(status_code=502, detail="investing price parse failed")

    price_text = price_el.get_text(strip=True)
    try:
        price = parse_number(price_text)
    except ValueError:
        raise HTTPException(status_code=502, detail="investing price invalid")

    name_el = soup.find(attrs={"data-test": "instrument-name"}) or soup.select_one("h1")
    name = name_el.get_text(strip=True) if name_el else quote.get("name") or ticker

    return {
        "provider": "investing",
        "ticker": ticker,
        "name": name,
        "price": float(price),
        "currency": "USD",
    }


def fetch_investing_search_quote(ticker: str, headers: dict | None = None) -> dict:
    headers = headers or {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json,text/plain,*/*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.investing.com/",
    }
    with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
        resp = client.post(INVESTING_SEARCH_URL, data={"search_text": ticker})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="investing search failed")

    try:
        payload = resp.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="investing search invalid")

    quotes = payload.get("quotes") or []
    if not quotes:
        raise HTTPException(status_code=404, detail="investing symbol not found")

    return quotes[0]
def fetch_fx_rate_to_krw(currency: str) -> float:
    if currency == "KRW":
        return 1.0
    fx_symbol = f"{currency}KRW=X"
    try:
        quote = fetch_yahoo_quote(fx_symbol)
        return float(quote["price"])
    except HTTPException:
        # Fallback to exchangerate.host for all currencies (including USD)
        url = f"https://api.exchangerate.host/latest?base={currency}&symbols=KRW"
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="fx fetch failed")
        data = resp.json()
        rate = data.get("rates", {}).get("KRW")
        if rate:
            return float(rate)

        # Final fallback: open.er-api.com
        url = f"https://open.er-api.com/v6/latest/{currency}"
        with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="fx fallback failed")
        data = resp.json()
        rate = data.get("rates", {}).get("KRW")
        if not rate:
            raise HTTPException(status_code=502, detail="fx rate missing")
        return float(rate)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        # Another process likely holds the write lock; skip for this worker.
        return

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                google_sub TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL,
                name TEXT,
                picture TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                asset_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                quantity INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        migration_id = "20250208_add_holdings_user_id"
        applied = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE id = ?",
            (migration_id,),
        ).fetchone()
        if not applied:
            columns = [
                row["name"]
                for row in conn.execute("PRAGMA table_info(holdings)").fetchall()
            ]
            if "user_id" not in columns:
                conn.execute("ALTER TABLE holdings ADD COLUMN user_id INTEGER")
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (id, applied_at) VALUES (?, ?)",
                (migration_id, datetime.now(timezone.utc).isoformat()),
            )
        conn.commit()
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")


def init_db() -> None:
    conn = get_db()
    try:
        apply_migrations(conn)
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "asset_type": row["asset_type"],
        "ticker": row["ticker"],
        "quantity": row["quantity"],
    }


class HoldingCreate(BaseModel):
    asset_type: str = Field(default="주식", min_length=1)
    ticker: str = Field(..., min_length=1)
    quantity: int = Field(0, ge=0)


class HoldingUpdate(BaseModel):
    asset_type: Optional[str] = None
    ticker: Optional[str] = None
    quantity: Optional[int] = None


class HoldingBulk(BaseModel):
    items: list[HoldingCreate]


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")

def require_user(request: Request) -> sqlite3.Row:
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="login required")
    conn = get_db()
    try:
        user = conn.execute(
            "SELECT id, google_sub, email, name, picture FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="user not found")
        return user
    finally:
        conn.close()


@app.get("/auth/login")
async def auth_login(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="google oauth not configured")
    redirect_uri = GOOGLE_REDIRECT_URI or str(request.url_for("auth_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="google oauth not configured")
    conn = get_db()
    try:
        user_count = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"]
        try:
            token = await oauth.google.authorize_access_token(request)
        except OAuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        user_info = token.get("userinfo")
        if not user_info:
            resp = await oauth.google.get("userinfo", token=token)
            user_info = resp.json()

        google_sub = user_info.get("sub")
        email = user_info.get("email")
        name = user_info.get("name") or ""
        picture = user_info.get("picture") or ""
        if not google_sub or not email:
            raise HTTPException(status_code=400, detail="google user info missing")

        existing = conn.execute(
            "SELECT id FROM users WHERE google_sub = ?",
            (google_sub,),
        ).fetchone()
        if existing:
            user_id = existing["id"]
            conn.execute(
                "UPDATE users SET email = ?, name = ?, picture = ? WHERE id = ?",
                (email, name, picture, user_id),
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO users (google_sub, email, name, picture, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (google_sub, email, name, picture, datetime.now(timezone.utc).isoformat()),
            )
            user_id = cur.lastrowid

        if user_count == 0:
            conn.execute(
                "UPDATE holdings SET user_id = ? WHERE user_id IS NULL",
                (user_id,),
            )

        conn.commit()
        request.session["user_id"] = user_id
        return RedirectResponse(url="/")
    finally:
        conn.close()


@app.post("/auth/logout")
def auth_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/auth/me")
def auth_me(request: Request):
    user = require_user(request)
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "picture": user["picture"],
    }


@app.get("/api/holdings")
def list_holdings(request: Request):
    user = require_user(request)
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, asset_type, ticker, quantity
            FROM holdings
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user["id"],),
        ).fetchall()
        holdings = []
        fx_cache: dict[str, float] = {}
        for row in rows:
            item = row_to_dict(row)
            error_detail = None
            try:
                if is_cash_ticker(item["ticker"]):
                    quote = {
                        "name": "현금/해당없음",
                        "price": 1.0,
                        "currency": "KRW",
                    }
                    price_krw = 1.0
                elif is_korean_ticker(item["ticker"]):
                    quote = fetch_naver_quote(item["ticker"])
                    price_krw = quote["price"]
                else:
                    quote = fetch_yahoo_quote(item["ticker"])
                    currency = quote.get("currency") or "USD"
                    if currency not in fx_cache:
                        fx_cache[currency] = fetch_fx_rate_to_krw(currency)
                    price_krw = quote["price"] * fx_cache[currency]
            except HTTPException as exc:
                quote = {"name": "조회실패", "price": None, "currency": "KRW"}
                price_krw = None
                error_detail = exc.detail

            amount = price_krw * item["quantity"] if price_krw is not None else None
            holdings.append(
                {
                    **item,
                    "name": quote["name"],
                    "price": price_krw,
                    "currency": "KRW",
                    "amount": amount,
                    "error": error_detail,
                }
            )
        return holdings
    finally:
        conn.close()


@app.get("/api/holdings_raw")
def list_holdings_raw(request: Request):
    user = require_user(request)
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, asset_type, ticker, quantity
            FROM holdings
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user["id"],),
        ).fetchall()
        return [row_to_dict(row) for row in rows]
    finally:
        conn.close()


@app.post("/api/holdings", status_code=201)
def create_holding(payload: HoldingCreate, request: Request):
    user = require_user(request)
    conn = get_db()
    try:
        asset_type = (payload.asset_type or "주식").strip()
        ticker = format_ticker(payload.ticker)
        if is_cash_ticker(ticker):
            asset_type = "예수금"
        cur = conn.execute(
            """
            INSERT INTO holdings (user_id, asset_type, ticker, quantity)
            VALUES (?, ?, ?, ?)
            """,
            (
                user["id"],
                asset_type,
                ticker,
                payload.quantity,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, asset_type, ticker, quantity FROM holdings WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


@app.post("/api/holdings/bulk_replace")
def bulk_replace(payload: HoldingBulk, request: Request):
    user = require_user(request)
    rows = []
    for item in payload.items:
        asset_type = (item.asset_type or "주식").strip()
        ticker = format_ticker(item.ticker)
        if not ticker:
            continue
        if is_cash_ticker(ticker):
            asset_type = "예수금"
        rows.append((user["id"], asset_type, ticker, item.quantity))

    conn = get_db()
    try:
        conn.execute("DELETE FROM holdings WHERE user_id = ?", (user["id"],))
        if rows:
            conn.executemany(
                """
                INSERT INTO holdings (user_id, asset_type, ticker, quantity)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()
    finally:
        conn.close()

    return {"saved": len(rows)}


@app.patch("/api/holdings/{holding_id}")
def update_holding(holding_id: int, payload: HoldingUpdate, request: Request):
    user = require_user(request)
    if payload.asset_type is None and payload.ticker is None and payload.quantity is None:
        raise HTTPException(status_code=400, detail="no fields to update")

    conn = get_db()
    try:
        existing = conn.execute(
            """
            SELECT id, asset_type, ticker, quantity
            FROM holdings
            WHERE id = ? AND user_id = ?
            """,
            (holding_id, user["id"]),
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="not found")

        updates = {}
        if payload.asset_type is not None:
            asset_type = payload.asset_type.strip()
            if not asset_type:
                raise HTTPException(status_code=400, detail="asset_type is required")
            updates["asset_type"] = asset_type
        if payload.ticker is not None:
            ticker = format_ticker(payload.ticker)
            if not ticker:
                raise HTTPException(status_code=400, detail="ticker is required")
            updates["ticker"] = ticker
        if payload.quantity is not None:
            if payload.quantity < 0:
                raise HTTPException(status_code=400, detail="quantity must be >= 0")
            updates["quantity"] = payload.quantity

        if updates:
            if "ticker" in updates and is_cash_ticker(updates["ticker"]):
                updates["asset_type"] = "예수금"
            fields = ", ".join(f"{key} = ?" for key in updates.keys())
            values = list(updates.values()) + [holding_id]
            conn.execute(f"UPDATE holdings SET {fields} WHERE id = ?", values)
            conn.commit()

        row = conn.execute(
            "SELECT id, asset_type, ticker, quantity FROM holdings WHERE id = ?",
            (holding_id,),
        ).fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


@app.delete("/api/holdings/{holding_id}", status_code=204)
def delete_holding(holding_id: int, request: Request):
    user = require_user(request)
    conn = get_db()
    try:
        cur = conn.execute(
            "DELETE FROM holdings WHERE id = ? AND user_id = ?",
            (holding_id, user["id"]),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="not found")
    finally:
        conn.close()


def parse_csv_text(text: str) -> list[tuple[str, str, int]]:
    reader = csv.DictReader(io.StringIO(text))
    required_cols = {"ticker", "quantity"}
    if not required_cols.issubset(set(reader.fieldnames or [])):
        raise HTTPException(status_code=400, detail="csv must include ticker,quantity")

    rows = []
    for row in reader:
        ticker = format_ticker(row.get("ticker", ""))
        if not ticker:
            continue
        qty_raw = row.get("quantity", "0")
        try:
            quantity = int(float(qty_raw))
        except ValueError:
            quantity = 0

        asset_type = "주식"
        if is_cash_ticker(ticker):
            asset_type = "예수금"

        rows.append((asset_type, ticker, quantity))
    return rows


@app.post("/api/import_csv_text")
async def import_csv_text(
    request: Request,
    replace: bool = Query(default=False),
    body: dict = Body(...),
):
    user = require_user(request)
    if not body or "csv" not in body:
        raise HTTPException(status_code=400, detail="csv text required")

    rows = parse_csv_text(str(body.get("csv", "")))
    rows = [(user["id"], asset_type, ticker, quantity) for asset_type, ticker, quantity in rows]
    conn = get_db()
    try:
        if replace:
            conn.execute("DELETE FROM holdings WHERE user_id = ?", (user["id"],))
        if rows:
            conn.executemany(
                """
                INSERT INTO holdings (user_id, asset_type, ticker, quantity)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()
    finally:
        conn.close()

    return {"imported": len(rows)}


@app.get("/api/quote")
def quote(
    ticker: str = Query(..., min_length=1),
):
    norm_ticker = format_ticker(ticker)
    if is_korean_ticker(norm_ticker):
        return fetch_naver_quote(norm_ticker)
    return fetch_yahoo_quote(norm_ticker)


@app.get("/api/quote_krw")
def quote_krw(
    ticker: str = Query(..., min_length=1),
):
    norm_ticker = format_ticker(ticker)
    if is_cash_ticker(norm_ticker):
        return {
            "ticker": norm_ticker,
            "name": "현금/해당없음",
            "price": 1.0,
            "currency": "KRW",
        }

    if is_korean_ticker(norm_ticker):
        quote = fetch_naver_quote(norm_ticker)
        return {
            "ticker": norm_ticker,
            "name": quote["name"],
            "price": quote["price"],
            "currency": "KRW",
        }

    errors = []
    try:
        quote = fetch_yahoo_quote(norm_ticker)
    except HTTPException as exc:
        errors.append(f"yahoo:{exc.detail}")
        try:
            quote = fetch_yahoo_quote_html(norm_ticker)
        except HTTPException as exc2:
            errors.append(f"yahoo_html:{exc2.detail}")
            try:
                quote = fetch_investing_quote(norm_ticker)
            except HTTPException as exc3:
                errors.append(f"investing:{exc3.detail}")
                try:
                    quote = fetch_stooq_quote(norm_ticker)
                except HTTPException as exc4:
                    errors.append(f"stooq:{exc4.detail}")
                    raise HTTPException(
                        status_code=502,
                        detail="; ".join(errors),
                    )
    currency = quote.get("currency") or "USD"
    try:
        fx_rate = fetch_fx_rate_to_krw(currency)
    except HTTPException as exc:
        raise HTTPException(status_code=502, detail=f"fx:{exc.detail}")
    price_krw = quote["price"] * fx_rate
    return {
        "ticker": norm_ticker,
        "name": NAME_OVERRIDES.get(norm_ticker, quote["name"]),
        "price": price_krw,
        "currency": "KRW",
    }
