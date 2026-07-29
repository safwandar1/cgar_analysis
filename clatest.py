import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ── Ticker name lookup ────────────────────────────────────────────────────────
TICKER_NAMES = {
    "VGT": "Vanguard Information Technology ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "QQQ": "Invesco QQQ Trust",
    "SCHG": "Schwab US Large-Cap Growth ETF",
    "SOXQ": "Invesco PHLX Semiconductor ETF",
    "QTUM": "Defiance Quantum ETF",
    "ITA": "iShares U.S. Aerospace & Defense ETF",
    "ABBV": "AbbVie Inc.",
    "LLY": "Eli Lilly and Company",
    "MCK": "McKesson Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "GS": "Goldman Sachs Group Inc.",
    "WMT": "Walmart Inc.",
    "NVDA": "NVIDIA Corporation",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "NFLX": "Netflix Inc.",
    "AMD": "Advanced Micro Devices Inc.",
    "BTC-USD": "Bitcoin USD",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corp.",
    "XOM": "Exxon Mobil Corp.",
    "CVX": "Chevron Corp.",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble Co.",
    "UNH": "UnitedHealth Group Inc.",
    "HD": "Home Depot Inc.",
    "COST": "Costco Wholesale Corp.",
    "AVGO": "Broadcom Inc.",
    "AMGN": "Amgen Inc.",
    "PFE": "Pfizer Inc.",
    "INTC": "Intel Corp.",
    "CSCO": "Cisco Systems Inc.",
    "IBM": "IBM Corp.",
    "QCOM": "Qualcomm Inc.",
    "TXN": "Texas Instruments Inc.",
    "AMAT": "Applied Materials Inc.",
    "MU": "Micron Technology Inc.",
    "SCHW": "Charles Schwab Corp.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "RTX": "RTX Corp.",
    "LMT": "Lockheed Martin Corp.",
    "NOC": "Northrop Grumman Corp.",
    "GD": "General Dynamics Corp.",
    "SPY": "SPDR S&P 500 ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "SCHD": "Schwab US Dividend Equity ETF",
    "VUG": "Vanguard Growth ETF",
    "ARKK": "ARK Innovation ETF",
}



# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAGR Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
# Streamlit does NOT re-run the script on its own — only on widget interaction
# or a fresh page load. Without this, a tab left open just keeps showing
# whatever was rendered the last time it actually executed, no matter how
# stale the underlying cached data becomes. This forces a rerun every 60s,
# which (combined with the ttl=3600 cache below) keeps prices reasonably
# current without the user needing to manually reload the page.
st_autorefresh(interval=60_000, key="data_refresh")

if "cache_cleared" not in st.session_state:
    st.cache_data.clear()
    st.session_state["cache_cleared"] = True

# ── Theme ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
.stApp { background-color: #f7fbf8; }
section[data-testid="stSidebar"] { background: linear-gradient(160deg, #0d3321 0%, #1a5c3a 100%); }
section[data-testid="stSidebar"] * { color: #d4f0e0 !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea {
    color: #0d3321 !important;
    background-color: #d4f0e0 !important;
    border: 1.5px solid #a8d5bc !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #0d3321 !important;
    background-color: #d4f0e0 !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] {
    background-color: #d4f0e0 !important;
    border-radius: 8px !important;
}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1.5px solid #c3e8d0;
    border-radius: 14px;
    padding: 18px 22px !important;
    box-shadow: 0 2px 12px rgba(0,80,40,0.06);
}
.ticker-tag {
    display: inline-block; background: #0d5c32; color: white;
    border-radius: 20px; padding: 3px 12px; font-size: 0.8rem; font-weight: 600; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ───────────────────────────────────────────────────────
def get_start_price(prices, year):
    yr_data = prices[prices.index.year == year]
    if not yr_data.empty:
        return float(yr_data.iloc[0]), yr_data.index[0]
    return None, None

def calculate_after_expense(initial_investment, years, cagr_percent, expense_ratio_decimal):
    """
    Calculate final value after annual fees with proper opportunity cost.
    Fees are deducted at the end of each year BEFORE the next year's growth.
    """
    if expense_ratio_decimal is None or expense_ratio_decimal <= 0:
        final_value = initial_investment * ((1 + cagr_percent/100) ** years)
        return final_value, 0

    value = initial_investment

    for year in range(years):
        # Grow the investment for the year
        value = value * (1 + cagr_percent/100)
        # Deduct the fee (opportunity cost - this fee doesn't grow in future years)
        value = value * (1 - expense_ratio_decimal)

    # Calculate total fees paid (opportunity cost)
    gross_value = initial_investment * ((1 + cagr_percent/100) ** years)
    total_fees = gross_value - value
    return value, total_fees

def get_expense_ratio(ticker_obj):
    """Get expense ratio - yfinance returns values like 0.09 for VGT (meaning 0.09%)"""
    try:
        info = ticker_obj.info

        # Try different field names that might contain the expense ratio
        expense_fields = ['annualReportExpenseRatio', 'expenseRatio', 'totalAnnualOperatingExpenses',
                         'managementExpenseRatio', 'grossExpenseRatio', 'netExpenseRatio']

        for field in expense_fields:
            value = info.get(field)
            if value is not None and isinstance(value, (int, float)):
                # yfinance returns the raw value which is already the percentage
                # For VGT: returns 0.09 which means 0.09%
                # For VTI: returns 0.03 which means 0.03%
                # So we need to convert to decimal by dividing by 100
                # 0.09% = 0.0009 as decimal
                decimal_value = value / 100
                return decimal_value

        return None
    except:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ticker_data(ticker, max_years):
    """
    Returns (prices, info) on success.
    Returns (None, error_message) on failure so the caller can show
    *why* a ticker failed instead of silently dropping it.
    """
    today = datetime.today()
    start_date = f"{today.year - max_years - 1}-01-01"
    try:
        data = yf.download(ticker, start=start_date, end=today.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if data.empty:
            return None, f"No data returned for '{ticker}' — check the ticker symbol."
        data.index = data.index.tz_localize(None)
        prices = data["Close"][ticker] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
        prices = prices.dropna()
        year_ago = prices.iloc[-252:] if len(prices) > 252 else prices

        # Fetch YTD directly from Yahoo Finance summary API — more reliable than price math
        try:
            import requests
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=ytd&interval=1d"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=5)
            chart = resp.json()["chart"]["result"][0]
            meta = chart["meta"]
            ytd_prices = chart["indicators"]["quote"][0]["close"]
            ytd_prices = [p for p in ytd_prices if p is not None]
            prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
            if ytd_prices and prev_close:
                # prev_close from YTD range = Dec 31 prior year close
                ytd_return = (ytd_prices[-1] / prev_close - 1) * 100
            else:
                ytd_return = None
        except Exception:
            # Fallback to price series math if API fails
            ytd_data = prices[prices.index.year == today.year]
            prior_year_data = prices[prices.index.year == today.year - 1]
            if not prior_year_data.empty and not ytd_data.empty:
                ytd_return = (float(ytd_data.iloc[-1]) / float(prior_year_data.iloc[-1]) - 1) * 100
            elif len(ytd_data) > 1:
                ytd_return = (float(ytd_data.iloc[-1]) / float(ytd_data.iloc[0]) - 1) * 100
            else:
                ytd_return = None

        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            mkt_cap = info.get("marketCap") or info.get("totalAssets")
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            expense_ratio_decimal = get_expense_ratio(ticker_obj)
            long_name = TICKER_NAMES.get(ticker) or info.get("longName") or info.get("shortName") or ""
        except:
            mkt_cap = None
            pe_ratio = None
            expense_ratio_decimal = None
            long_name = TICKER_NAMES.get(ticker, "")

        p_info = {
            "current": float(prices.iloc[-1]),
            "hi52": float(year_ago.max()),
            "lo52": float(year_ago.min()),
            "ytd_return": ytd_return,
            "mkt_cap": mkt_cap,
            "pe_ratio": pe_ratio,
            "expense_ratio": expense_ratio_decimal,
            "long_name": long_name,
            "fetched_at": today.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return prices, p_info
    except Exception as e:
        return None, f"Error fetching '{ticker}': {e}"

def compute_cagr_df(prices, max_years):
    today = datetime.today()
    curr_val = float(prices.iloc[-1])
    results = []
    for n in range(1, max_years + 1):
        target_year = today.year - n
        start_val, start_dt = get_start_price(prices, target_year)
        if start_val and start_dt:
            days = (prices.index[-1] - start_dt).days
            if days > 0:
                years_elapsed = days / 365.25
                cagr = (curr_val / start_val)**(1/years_elapsed) - 1
                results.append({"n": n, "year": target_year, "cagr": round(cagr * 100, 2)})
    return pd.DataFrame(results)

def compute_range_cagr(prices, n_start, n_end):
    today = datetime.today()

    if n_start == 0:
        val_to = float(prices.iloc[-1])
        dt_to  = prices.index[-1]
    else:
        year_to = today.year - n_start
        val_to, dt_to = get_start_price(prices, year_to)

    if n_end == 0:
        val_from = float(prices.iloc[-1])
        dt_from  = prices.index[-1]
    else:
        year_from = today.year - n_end
        val_from, dt_from = get_start_price(prices, year_from)

    if val_from and val_to and dt_from is not None and dt_to is not None:
        days = (dt_to - dt_from).days
        if days <= 0:
            return None, val_from, val_to, dt_from, dt_to
        years_elapsed = days / 365.25
        cagr = (val_to / val_from)**(1/years_elapsed) - 1
        return round(cagr * 100, 2), val_from, val_to, dt_from, dt_to
    return None, None, None, None, None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📈 Settings")
    t_input = st.text_input("Tickers", "VGT")
    investment = st.number_input("Investment ($)", value=100000)
    lookback = st.slider("Max Lookback", 3, 20, 15)
    st.markdown("**CAGR Range** (years ago)")
    st.caption("Drag both ends to compare any two periods")
    horizon_range = st.select_slider(
        "From → To (years ago)",
        options=list(range(0, lookback + 1)),
        value=(min(5, lookback), min(10, lookback)),
    )
    horizon_start, horizon_end = min(horizon_range), max(horizon_range)
    st.divider()
    st.caption(f"🔄 Auto-refreshing every 60s · Last script run: {datetime.now().strftime('%H:%M:%S')}")

# ── Main Logic ────────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in t_input.split(",") if t.strip()]
all_data = {}
fetch_errors = {}

for t in tickers:
    p, info = fetch_ticker_data(t, lookback)
    if p is not None:
        all_data[t] = {"prices": p, "info": info, "cagr_df": compute_cagr_df(p, lookback)}
    else:
        fetch_errors[t] = info  # info holds the error message string in this branch

if fetch_errors:
    for t, msg in fetch_errors.items():
        st.error(f"⚠️ **{t}**: {msg}")

if not all_data:
    st.warning("No data found. Check ticker symbols.")
    st.stop()

st.title("Portfolio CAGR Analyzer")
st.markdown("".join([f'<span class="ticker-tag">{t}</span>' for t in all_data]), unsafe_allow_html=True)

# ── Create Tabs AFTER all data is loaded ──────────────────────────────────────
tabs = st.tabs(["📊 CAGR Hero", "📅 Annual Returns", "💰 Growth Chart", "🚀 Future Comparison", "⚡ Volatility & Sharpe", "📉 Drawdown", "🔍 Sharpe Screener"])

def fmt_mkt_cap(val):
    if val is None: return "N/A"
    if val >= 1e12: return f"${val/1e12:.2f}T"
    if val >= 1e9:  return f"${val/1e9:.2f}B"
    if val >= 1e6:  return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"

today_year = datetime.today().year

# ── Tab 0: CAGR Hero ──────────────────────────────────────────────────────────
with tabs[0]:
    for tkr, d in all_data.items():
        prices = d["prices"]
        pi = d["info"]
        _long = pi.get("long_name", "")
        _title = f"{tkr} — {_long}" if _long and _long != tkr else tkr
        st.subheader(_title)
        st.caption(f"Data as of {pi.get('fetched_at', 'unknown')}")

        ytd = pi.get("ytd_return")
        ytd_str = f"{ytd:+.2f}%" if ytd is not None else "N/A"
        ytd_delta = f"{ytd:+.2f}%" if ytd is not None else None

        pe = pi.get("pe_ratio")
        pe_str = f"{pe:.1f}x" if pe is not None else "N/A"

        expense_decimal = pi.get("expense_ratio")
        expense_str = f"{expense_decimal*100:.3f}%" if expense_decimal is not None else "N/A"

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Current Price", f"${pi['current']:,.2f}")
        col2.metric("52W High", f"${pi['hi52']:,.2f}")
        col3.metric("52W Low", f"${pi['lo52']:,.2f}")
        col4.metric("Mkt Cap / AUM", fmt_mkt_cap(pi.get("mkt_cap")))
        col5.metric("YTD Return", ytd_str, delta=ytd_delta)
        col6.metric("Expense Ratio", expense_str)

        col_a, col_b = st.columns(2)

        span_years = horizon_end - horizon_start
        range_label = f"{today_year - horizon_end} → {today_year - horizon_start}"
        r_cagr, r_from, r_to, r_dt_from, r_dt_to = compute_range_cagr(prices, horizon_start, horizon_end)

        with col_a:
            if r_cagr is not None:
                color = "#0d5c32" if r_cagr >= 0 else "#b91c1c"
                st.markdown(f"""
                <div style="background:white; border:1.5px solid #c3e8d0; border-radius:18px; padding:25px; margin:10px 0;">
                    <div style="font-size:0.7rem; text-transform:uppercase; color:#4a7c5f;">📅 Historical CAGR · {range_label}</div>
                    <div style="font-size:2.6rem; font-weight:800; color:{color};">{r_cagr:+.2f}%</div>
                    <div style="margin-top:12px; font-size:0.82rem; color:#555;">
                        <b>From</b> ${r_from:,.2f} ({r_dt_from.strftime('%b %Y')}) → <b>To</b> ${r_to:,.2f} ({r_dt_to.strftime('%b %Y')})
                    </div>
                    <div style="margin-top:8px; font-size:0.8rem; color:#888;">{span_years}-year window</div>
                </div>
                """, unsafe_allow_html=True)
            elif r_from is not None:
                st.info(f"⚠️ Same period on both ends ({range_label}). Drag the slider to create a range of at least 1 year.")
            else:
                st.warning(f"Not enough data for {range_label} range.")

        with col_b:
            if r_cagr is not None:
                future_val = investment * ((1 + r_cagr / 100) ** span_years)
                future_year = today_year + span_years
                color2 = "#0d5c32" if r_cagr >= 0 else "#b91c1c"

                # Calculate after expense with proper opportunity cost
                after_expense_val, drag_amount = calculate_after_expense(investment, span_years, r_cagr, expense_decimal)
                expense_display = f"{expense_decimal*100:.3f}%" if expense_decimal else None

                st.markdown(f"""
                <div style="background:white; border:1.5px solid #bde0cc; border-radius:18px; padding:25px; margin:10px 0;">
                    <div style="font-size:0.7rem; text-transform:uppercase; color:#4a7c5f;">🚀 If Same CAGR Repeats · {today_year} → {future_year}</div>
                    <div style="font-size:2.6rem; font-weight:800; color:{color2};">{r_cagr:+.2f}%</div>
                    <div style="margin-top:12px; font-size:0.82rem; color:#555;">
                        <b>${investment:,.0f}</b> today → <b>${future_val:,.0f}</b> by {future_year}
                    </div>
                    <div style="margin-top:4px; font-size:0.8rem; color:#888;">
                        Gain: +${future_val - investment:,.0f} over {span_years} years
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Expense info using st.info (clean and simple)
                if expense_decimal is not None and expense_decimal > 0:
                    st.info(f"📉 **After {expense_display} expense ratio:** ${after_expense_val:,.0f}  \n\n*Lost ${drag_amount:,.0f} to fees (opportunity cost) over {span_years} years*")
        st.divider()

# ── Tab 1: Annual Returns ──────────────────────────────────────────────────────
with tabs[1]:
    for tkr, d in all_data.items():
        prices = d["prices"]
        yrs = sorted(prices.index.year.unique())
        ann_rets = []
        for y in yrs:
            yr_data = prices[prices.index.year == y]
            if len(yr_data) > 1:
                ret = (yr_data.iloc[-1] / yr_data.iloc[0] - 1) * 100
                ann_rets.append({"Year": str(y), "Return": ret})

        if ann_rets:
            fig = px.bar(pd.DataFrame(ann_rets), x="Year", y="Return", title=f"{tkr} Annual Returns",
                        color="Return", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Growth Chart ────────────────────────────────────────────────────────
with tabs[2]:
    fig_g = go.Figure()
    for tkr, d in all_data.items():
        norm = (d["prices"] / d["prices"].iloc[0]) * investment
        fig_g.add_trace(go.Scatter(x=norm.index, y=norm.values, name=tkr))
    fig_g.update_layout(title="Growth Over Time", yaxis_tickprefix="$", template="plotly_white")
    st.plotly_chart(fig_g, use_container_width=True)

# ── Tab 3: Future Comparison ───────────────────────────────────────────────────
with tabs[3]:
    st.subheader(f"Future Growth Comparison (${investment:,.0f} Investment)")
    span_years = horizon_end - horizon_start
    range_label = f"{today_year - horizon_end} → {today_year - horizon_start}"
    future_year = today_year + span_years
    st.info(f"📅 Historical: {range_label} &nbsp;|&nbsp; 🚀 Forward projection: {today_year} → {future_year}")

    for tkr, d in all_data.items():
        prices = d["prices"]
        pi = d["info"]
        expense_decimal = pi.get("expense_ratio")
        r_cagr, _, _, _, _ = compute_range_cagr(prices, horizon_start, horizon_end)

        if r_cagr is not None:
            proj_historical = investment * ((1 + r_cagr/100)**span_years)
            proj_forward = investment * ((1 + r_cagr/100)**span_years)

            # Calculate after expense with proper opportunity cost
            after_expense_hist, drag_hist = calculate_after_expense(investment, span_years, r_cagr, expense_decimal)
            after_expense_fwd, drag_fwd = calculate_after_expense(investment, span_years, r_cagr, expense_decimal)

            expense_display = f"{expense_decimal*100:.3f}%" if expense_decimal else None

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label=f"📅 {tkr} Historical ({range_label})",
                        value=f"${proj_historical:,.0f}",
                        delta=f"{r_cagr:+.2f}% CAGR"
                    )
                    if expense_decimal:
                        st.caption(f"💰 After {expense_display} fees: ${after_expense_hist:,.0f}")
                        st.caption(f"📉 Lost ${drag_hist:,.0f} to fees")

                with col2:
                    st.metric(
                        label=f"🚀 {tkr} Forward ({today_year}→{future_year})",
                        value=f"${proj_forward:,.0f}",
                        delta=f"Same {r_cagr:+.2f}% rate"
                    )
                    if expense_decimal:
                        st.caption(f"💰 After {expense_display} fees: ${after_expense_fwd:,.0f}")
                        st.caption(f"📉 Lost ${drag_fwd:,.0f} to fees")

# ── Tab 4: Volatility & Sharpe ─────────────────────────────────────────────────
with tabs[4]:
    RISK_FREE_RATE = 0.045  # ~4.5% approximate current T-bill rate
    st.caption(f"Risk-free rate assumed: {RISK_FREE_RATE*100:.1f}% (approx. T-bill). Sharpe = (CAGR - Rf) / Annualized Volatility.")

    summary_rows = []
    fig_vol = go.Figure()

    for tkr, d in all_data.items():
        prices = d["prices"]

        # Daily returns
        daily_ret = prices.pct_change().dropna()

        # Rolling 30-day annualized volatility
        rolling_vol = daily_ret.rolling(30).std() * np.sqrt(252) * 100
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol.values,
            name=tkr, mode="lines", line=dict(width=1.5)
        ))

        # Overall stats
        ann_vol = float(daily_ret.std() * np.sqrt(252))
        curr = float(prices.iloc[-1])
        start = float(prices.iloc[0])
        days_total = (prices.index[-1] - prices.index[0]).days
        years_total = days_total / 365.25
        cagr_full = (curr / start) ** (1 / years_total) - 1
        sharpe = (cagr_full - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else None

        # Best / worst single day
        best_day = float(daily_ret.max()) * 100
        worst_day = float(daily_ret.min()) * 100

        # % of positive days
        pct_positive = (daily_ret > 0).sum() / len(daily_ret) * 100

        summary_rows.append({
            "Ticker": tkr,
            "CAGR (full)": f"{cagr_full*100:.2f}%",
            "Ann. Volatility": f"{ann_vol*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}" if sharpe else "N/A",
            "Best Day": f"+{best_day:.2f}%",
            "Worst Day": f"{worst_day:.2f}%",
            "% Positive Days": f"{pct_positive:.1f}%",
        })

    # Rolling vol chart
    fig_vol.update_layout(
        title="Rolling 30-Day Annualized Volatility",
        yaxis_title="Volatility (%)",
        yaxis_ticksuffix="%",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Summary table
    st.subheader("Risk & Return Summary")
    st.dataframe(
        pd.DataFrame(summary_rows).set_index("Ticker"),
        use_container_width=True,
    )

    # Per-ticker Sharpe bar chart
    sharpe_vals = []
    for row in summary_rows:
        try:
            sharpe_vals.append({"Ticker": row["Ticker"], "Sharpe": float(row["Sharpe Ratio"])})
        except (ValueError, TypeError):
            pass

    if sharpe_vals:
        df_sh = pd.DataFrame(sharpe_vals)
        colors_sh = ["#0d5c32" if v >= 1 else "#b47a00" if v >= 0 else "#b91c1c" for v in df_sh["Sharpe"]]
        fig_sh = go.Figure(go.Bar(
            x=df_sh["Ticker"], y=df_sh["Sharpe"],
            marker_color=colors_sh,
            text=[f"{v:.2f}" for v in df_sh["Sharpe"]],
            textposition="outside",
        ))
        fig_sh.update_layout(
            title="Sharpe Ratio by Ticker  (green ≥ 1.0 · amber ≥ 0 · red < 0)",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig_sh, use_container_width=True)


# ── Tab 5: Drawdown ────────────────────────────────────────────────────────────
with tabs[5]:
    fig_dd = go.Figure()
    dd_summary = []

    for tkr, d in all_data.items():
        prices = d["prices"]

        # Rolling drawdown from peak
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max * 100

        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            name=tkr, mode="lines", line=dict(width=1.5),
            fill="tozeroy", opacity=0.15,
        ))

        max_dd = float(drawdown.min())
        max_dd_date = drawdown.idxmin()

        # Recovery: first date after max_dd_date where drawdown >= -1%
        after = drawdown[drawdown.index > max_dd_date]
        recovered = after[after >= -1]
        recovery_date = recovered.index[0] if not recovered.empty else None
        recovery_days = (recovery_date - max_dd_date).days if recovery_date else None

        # Current drawdown from ATH
        current_dd = float(drawdown.iloc[-1])

        dd_summary.append({
            "Ticker": tkr,
            "Max Drawdown": f"{max_dd:.2f}%",
            "Max DD Date": max_dd_date.strftime("%b %Y"),
            "Days to Recover": str(recovery_days) if recovery_days else "Not yet",
            "Current DD from ATH": f"{current_dd:.2f}%",
        })

    fig_dd.update_layout(
        title="Drawdown from All-Time High",
        yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Drawdown Summary")
    st.dataframe(
        pd.DataFrame(dd_summary).set_index("Ticker"),
        use_container_width=True,
    )

    # Per-ticker max drawdown bar
    df_dd_bar = pd.DataFrame([{"Ticker": r["Ticker"], "Max DD": float(r["Max Drawdown"].replace("%",""))} for r in dd_summary])
    fig_dd_bar = go.Figure(go.Bar(
        x=df_dd_bar["Ticker"],
        y=df_dd_bar["Max DD"],
        marker_color="#b91c1c",
        text=[f"{v:.1f}%" for v in df_dd_bar["Max DD"]],
        textposition="outside",
    ))
    fig_dd_bar.update_layout(
        title="Max Drawdown by Ticker",
        yaxis_title="Max Drawdown (%)",
        yaxis_ticksuffix="%",
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig_dd_bar, use_container_width=True)

# ── Tab 6: Sharpe Screener ─────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("S&P 500 Sharpe Ratio Screener")
    st.caption("Fetches live data for a large batch of S&P 500 stocks and ranks by Sharpe ratio. Takes ~2–4 min on first run — results are cached for 1 hour.")

    # ── Universe ──────────────────────────────────────────────────────────────
    SP500_TICKERS = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","AVGO","JPM",
        "LLY","V","UNH","XOM","MA","COST","HD","PG","JNJ","ABBV",
        "BAC","MRK","CVX","CRM","NFLX","AMD","PEP","TMO","ACN","ADBE",
        "WMT","LIN","MCD","ABT","CSCO","TXN","DHR","AMGN","PM","NEE",
        "RTX","QCOM","INTU","SPGI","HON","CAT","IBM","GE","AMAT","ISRG",
        "GS","BLK","SYK","VRTX","AXP","T","VZ","REGN","PLD","MS",
        "SCHW","NOW","GILD","MDT","SO","ADP","CI","TJX","DUK","CME",
        "BSX","MO","ZTS","AON","MDLZ","PH","ETN","SLB","ICE","ELV",
        "USB","KLAC","HCA","WM","ITW","MCK","APD","NOC","EMR","EOG",
        "MCO","CL","ADI","LRCX","PSX","MPC","FCX","F","GM","NXPI",
        "KO","PFE","LOW","UPS","DE","MMM","FDX","NSC","CSX","UNP",
        "PYPL","INTC","BA","GD","LMT","WFC","C","COF","AIG","PRU",
        "D","EXC","XEL","ED","WEC","ES","AWK","FE","PEG","ETR",
        "AMT","CCI","EQIX","PSA","SPG","O","WELL","DLR","AVB","EQR",
    ]

    SCREENER_RF = 0.045

    col_sc1, col_sc2, col_sc3 = st.columns(3)
    with col_sc1:
        screen_period = st.selectbox("Lookback period", ["1Y", "2Y", "3Y", "5Y"], index=0)
    with col_sc2:
        min_sharpe = st.number_input("Min Sharpe to show", value=1.0, step=0.1)
    with col_sc3:
        top_n = st.number_input("Show top N results", value=30, step=5, min_value=5, max_value=150)

    period_map = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5}
    screen_years = period_map[screen_period]

    @st.cache_data(show_spinner=False, ttl=3600)
    def fetch_screener_data(tickers, years):
        today = datetime.today()
        start = f"{today.year - years - 1}-01-01"
        results = []
        # Batch download all tickers at once — much faster than one by one
        try:
            raw = yf.download(
                tickers, start=start,
                end=today.strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True
            )
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                closes = raw[["Close"]]
                closes.columns = tickers[:1]
        except Exception:
            return pd.DataFrame()

        for tkr in tickers:
            try:
                if tkr not in closes.columns:
                    continue
                prices = closes[tkr].dropna()
                if len(prices) < 60:
                    continue
                # Trim to requested period
                cutoff = today - pd.DateOffset(years=years)
                prices = prices[prices.index >= cutoff]
                if len(prices) < 60:
                    continue

                daily_ret = prices.pct_change().dropna()
                ann_vol = float(daily_ret.std() * np.sqrt(252))
                if ann_vol == 0:
                    continue

                cagr = (float(prices.iloc[-1]) / float(prices.iloc[0])) ** (252 / len(prices)) - 1
                sharpe = (cagr - SCREENER_RF) / ann_vol

                best_day = float(daily_ret.max()) * 100
                worst_day = float(daily_ret.min()) * 100
                pct_pos = (daily_ret > 0).sum() / len(daily_ret) * 100

                # Max drawdown
                roll_max = prices.cummax()
                dd = (prices - roll_max) / roll_max
                max_dd = float(dd.min()) * 100

                results.append({
                    "Ticker": tkr,
                    "CAGR": round(cagr * 100, 2),
                    "Volatility": round(ann_vol * 100, 2),
                    "Sharpe": round(sharpe, 2),
                    "Max Drawdown": round(max_dd, 2),
                    "Best Day %": round(best_day, 2),
                    "Worst Day %": round(worst_day, 2),
                    "% Pos Days": round(pct_pos, 1),
                })
            except Exception:
                continue

        df = pd.DataFrame(results).sort_values("Sharpe", ascending=False).reset_index(drop=True)
        return df

    run_screen = st.button("🔍 Run Screener", type="primary")

    if run_screen:
        with st.spinner(f"Fetching {len(SP500_TICKERS)} tickers for {screen_period} period… this takes ~1–2 min on first run."):
            df_screen = fetch_screener_data(tuple(SP500_TICKERS), screen_years)

        if df_screen.empty:
            st.warning("No data returned. Try again.")
        else:
            df_filtered = df_screen[df_screen["Sharpe"] >= min_sharpe].head(int(top_n)).copy()

            # ── Summary cards ──────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers screened", len(df_screen))
            c2.metric(f"Sharpe ≥ {min_sharpe:.1f}", len(df_screen[df_screen["Sharpe"] >= min_sharpe]))
            c3.metric("Highest Sharpe", f"{df_screen['Sharpe'].max():.2f} ({df_screen.iloc[0]['Ticker']})")
            c4.metric("Median Sharpe", f"{df_screen['Sharpe'].median():.2f}")

            # ── Top N bar chart ────────────────────────────────────────────
            bar_colors = ["#0d5c32" if v >= 2 else "#b47a00" if v >= 1 else "#b91c1c"
                          for v in df_filtered["Sharpe"]]
            fig_sc = go.Figure(go.Bar(
                x=df_filtered["Ticker"],
                y=df_filtered["Sharpe"],
                marker_color=bar_colors,
                text=[f"{v:.2f}" for v in df_filtered["Sharpe"]],
                textposition="outside",
            ))
            fig_sc.update_layout(
                title=f"Top {len(df_filtered)} by Sharpe Ratio ({screen_period}, Sharpe ≥ {min_sharpe})  |  green ≥ 2.0 · amber ≥ 1.0",
                yaxis_title="Sharpe Ratio",
                template="plotly_white",
                showlegend=False,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            # ── Scatter: CAGR vs Volatility, sized by Sharpe ───────────────
            fig_scatter = px.scatter(
                df_filtered,
                x="Volatility", y="CAGR",
                size=df_filtered["Sharpe"].clip(lower=0.1),
                color="Sharpe",
                text="Ticker",
                color_continuous_scale="RdYlGn",
                title="CAGR vs Volatility (bubble size = Sharpe ratio)",
                labels={"Volatility": "Annualized Volatility (%)", "CAGR": "CAGR (%)"},
            )
            fig_scatter.update_traces(textposition="top center", textfont_size=10)
            fig_scatter.update_layout(template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)

            # ── Full table ─────────────────────────────────────────────────
            st.subheader(f"Full results — top {len(df_filtered)} stocks")

            def color_sharpe(val):
                if val >= 2.0:   return "background-color:#d4f0e0; color:#0d5c32; font-weight:600"
                elif val >= 1.0: return "background-color:#fff8e0; color:#b47a00; font-weight:600"
                else:            return "background-color:#fde8e8; color:#b91c1c"

            styled = (
                df_filtered
                .rename(columns={
                    "CAGR": "CAGR %", "Volatility": "Vol %",
                    "Max Drawdown": "Max DD %",
                    "Best Day %": "Best Day", "Worst Day %": "Worst Day",
                })
                .style
                .map(color_sharpe, subset=["Sharpe"])
                .format({
                    "CAGR %": "{:.2f}%", "Vol %": "{:.2f}%",
                    "Sharpe": "{:.2f}",
                    "Max DD %": "{:.2f}%",
                    "Best Day": "{:.2f}%", "Worst Day": "{:.2f}%",
                    "% Pos Days": "{:.1f}%",
                })
            )
            st.dataframe(styled, use_container_width=True, height=600)

            # ── Download button ────────────────────────────────────────────
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                "⬇ Download results as CSV",
                data=csv,
                file_name=f"sharpe_screener_{screen_period}_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
    else:
        st.info("👆 Click **Run Screener** to fetch live data and rank all tickers by Sharpe ratio.")
