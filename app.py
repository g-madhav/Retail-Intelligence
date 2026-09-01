import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RetailIQ", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# CSS FOR ONE-PAGE DYNAMIC DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --blue: #2563eb; --blue-lt: #eff6ff; --blue-dk: #1e40af;
    --teal: #059669; --teal-lt: #ecfdf5;
    --amber: #d97706; --amber-lt: #fffbeb;
    --purple: #7c3aed; --purple-lt: #f5f3ff;
    --coral: #ef4444; --coral-lt: #fef2f2;
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --text-main: #020617;     
    --text-sub: #1e293b;      
    --text-muted: #475569;
}

/* Base Body */
html, body, .stApp {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
    background: linear-gradient(-45deg, #f8fafc, #eff6ff, #f8fafc, #f5f3ff);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    scroll-behavior: smooth;
}
#MainMenu, footer, header, [data-testid="stSidebar"], [data-testid="stDecoration"], [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 1440px !important; margin: 0 auto !important;}

@keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes slideUp { 0% { opacity: 0; transform: translateY(40px); } 100% { opacity: 1; transform: translateY(0); } }

/* Navbar */
.nav {
    position: sticky; top: 0; z-index: 1000;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(203, 213, 225, 0.6);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 3rem; height: 75px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.04);
}
.brand { font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 800; display: flex; align-items: center; gap: 12px; color: var(--text-main); letter-spacing: -0.02em; }
.brand-dot { width: 16px; height: 16px; border-radius: 50%; background: linear-gradient(135deg, var(--blue), var(--purple)); }
.nav-tabs { display: flex; gap: 8px; }
.tab-btn { background: transparent; border: none; cursor: pointer; font-size: 16px; font-weight: 700; padding: 10px 20px; border-radius: 99px; color: var(--text-muted); font-family: 'Inter', sans-serif; transition: all 0.3s ease; text-decoration: none; }
.tab-btn:hover { color: var(--blue); background: rgba(37,99,235,0.08); transform: translateY(-2px); }

/* Typography */
.section { padding: 4rem 3rem 2rem 3rem; scroll-margin-top: 75px; }
.section-title { font-family: 'Outfit', sans-serif !important; font-size: 2.8rem !important; font-weight: 800 !important; margin-bottom: 0.5rem !important; letter-spacing: -0.03em !important; background: linear-gradient(135deg, #020617 0%, #2563eb 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; line-height: 1.2 !important; }
.section-sub { font-size: 1.3rem !important; color: var(--text-sub) !important; line-height: 1.7 !important; margin-bottom: 2.5rem !important; max-width: 900px !important; font-weight: 500 !important; }

/* Insight Banners */
.insight { background: linear-gradient(135deg, var(--blue-lt) 0%, #ffffff 100%); border-left: 6px solid var(--blue); border-radius: 0 16px 16px 0; padding: 2rem; margin-bottom: 2.5rem; box-shadow: 0 4px 15px rgba(37,99,235,0.05); }
.insight-title { font-size: 17px; font-weight: 800; color: var(--blue-dk); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.08em; }
.insight-text { font-size: 1.15rem; color: var(--text-main); line-height: 1.7; font-weight: 500; }

/* KPIs */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
.kpi { background: var(--bg-card); border-radius: 20px; padding: 1.75rem; border: 1px solid #cbd5e1; box-shadow: 0 4px 20px rgba(0,0,0,0.03); transition: all 0.3s ease; animation: slideUp 0.8s ease backwards; }
.kpi:hover { transform: translateY(-8px); box-shadow: 0 20px 30px -5px rgba(37,99,235,0.1); border-color: rgba(37, 99, 235, 0.3); }
.kpi-label { font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-sub); margin-bottom: 10px; }
.kpi-val { font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 900; margin-bottom: 8px; color: var(--text-main); line-height: 1; letter-spacing: -0.03em;}
.kpi-sub { font-size: 15px; color: var(--text-muted); font-weight: 600; }
.badge { display: inline-flex; align-items: center; gap: 4px; font-size: 13px; padding: 6px 14px; border-radius: 99px; margin-top: 12px; font-weight: 700; }
.badge-blue { background: var(--blue-lt); color: var(--blue-dk); }
.badge-green { background: var(--teal-lt); color: var(--teal); }

/* HTML Cards inside Streamlit native containers */
.card-title { font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 900; margin-bottom: 5px; color: var(--text-main); letter-spacing: -0.02em; }
.card-desc { font-size: 1.1rem; color: var(--text-muted); margin-bottom: 1.5rem; line-height: 1.6; font-weight: 500; }

/* ═════ STREAMLIT NATIVE CARDS & SCROLL ANIMATION ═════ */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.03) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    animation: slideUp 0.8s ease backwards !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 20px 40px rgba(37,99,235,0.08) !important;
    border-color: rgba(37, 99, 235, 0.3) !important;
}

/* Stagger Animation Delays */
div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(1) { animation-delay: 0.1s !important; }
div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(2) { animation-delay: 0.2s !important; }
div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(3) { animation-delay: 0.3s !important; }
div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(4) { animation-delay: 0.4s !important; }


/* ═════ STREAMLIT WIDGET POLISH ═════ */
div[data-testid="stWidgetLabel"] p { font-size: 14px !important; font-weight: 800 !important; color: #0f172a !important; margin-bottom: 8px !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }

/* Launch AI Button */
div[data-testid="stFormSubmitButton"] button, div[data-testid="stFormSubmitButton"] button:active, div[data-testid="stFormSubmitButton"] button:focus {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important; border-radius: 14px !important; 
    min-height: 65px !important; width: 100% !important;
    margin-top: 25px !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 10px 25px rgba(37,99,235,0.3) !important; 
}
div[data-testid="stFormSubmitButton"] button p {
    color: #ffffff !important; font-family: 'Outfit', sans-serif !important; 
    font-size: 1.5rem !important; font-weight: 900 !important;
    text-transform: uppercase !important; letter-spacing: 0.05em !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-5px) scale(1.02) !important;
    box-shadow: 0 20px 40px rgba(124,58,237,0.4) !important; 
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HIGH VISIBILITY PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
def apply_plotly_theme(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#020617", size=14), title=None,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="#cbd5e1", zerolinecolor="#cbd5e1", title_font=dict(size=15, color="#0f172a", family="Outfit"), tickfont=dict(size=14, color="#1e293b")),
        yaxis=dict(gridcolor="#cbd5e1", zerolinecolor="#cbd5e1", title_font=dict(size=15, color="#0f172a", family="Outfit"), tickfont=dict(size=14, color="#1e293b")),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, font=dict(size=14, color="#0f172a", family="Outfit")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", font_size=16, font_family="Inter", bordercolor="#2563eb", font_color="#020617")
    )
    fig.update_traces(line=dict(width=4), marker=dict(size=10, line=dict(width=1, color="white")), selector=dict(type='scatter'))
    return fig

C_BLU = "#2563eb"; C_PUR = "#7c3aed"; C_TEA = "#059669"; C_COR = "#ef4444"; C_AMB = "#d97706"

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    return pd.read_csv(
        BASE_DIR / "retail_engineered.csv"
    )

@st.cache_resource
def load_model():
    return joblib.load(
        BASE_DIR / "demand_forecasting_model.pkl"
    )

try:
    df = load_data()
    model = load_model()

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.exception(e)
    st.stop()

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ══════════════════════════════════════════════════════════════════════════════
#  NAVBAR 
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nav">
  <div class="brand"><div class="brand-dot"></div>RetailIQ</div>
  <div class="nav-tabs">
    <a href="#overview" class="tab-btn">Overview</a>
    <a href="#demand-dynamics" class="tab-btn">Demand</a>
    <a href="#price-elasticity" class="tab-btn">Pricing</a>
    <a href="#ai-predictor" class="tab-btn">AI Predictor</a>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: OVERVIEW 
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="overview" class="section">', unsafe_allow_html=True)
st.markdown("""
<div class="section-title">Retail Intelligence & Demand Forecasting</div>
<div class="section-sub">Machine learning-powered demand forecasting for a Brazilian e-commerce retailer. A Gradient Boosting Regressor trained on historical sales data explains 63.6% of variance in order quantities, allowing for hyper-accurate inventory planning.</div>
<div class="insight">
    <div class="insight-title">System Architecture & Methodology</div>
    <div class="insight-text">The data pipeline originates from <b>retail_price.csv</b>, ingested into a local MySQL database via SQLAlchemy. Intensive EDA and feature engineering (One-Hot Encoding, Temporal extraction, Lead/Lag features) prepare the data for the Gradient Boosting Regressor. Optimized with 200 estimators and a 0.05 learning rate, it out-performs baseline RF models.</div>
</div>
""", unsafe_allow_html=True)

cat_cols = [c for c in df.columns if c.startswith("product_category_name_")]
cat_demand = {col.replace("product_category_name_","").replace("_"," ").title(): df.loc[df[col]==1,"qty"].sum() for col in cat_cols}
cat_df = pd.DataFrame({"Category": list(cat_demand.keys()), "Demand": list(cat_demand.values())}).sort_values("Demand", ascending=False).reset_index(drop=True)
top_cat = cat_df.iloc[0]["Category"] if not cat_df.empty else "N/A"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi"><div class="kpi-label">Total Transactions</div><div class="kpi-val">{len(df):,}</div><div class="kpi-sub">Historical sales records</div><div class="badge badge-blue">2017–2018 Data</div></div>
    <div class="kpi"><div class="kpi-label">Top Category</div><div class="kpi-val" style="font-size:2rem;line-height:1.2;margin-top:6px">{top_cat}</div><div class="kpi-sub">Highest volume driver</div><div class="badge badge-green">Strongest Performer</div></div>
    <div class="kpi"><div class="kpi-label">Avg Order Size</div><div class="kpi-val">{df['qty'].mean():.1f}</div><div class="kpi-sub">Units per transaction</div><div class="badge badge-blue">Max recorded: {df['qty'].max():.0f}</div></div>
    <div class="kpi"><div class="kpi-label">Model Accuracy (R²)</div><div class="kpi-val">0.636</div><div class="kpi-sub">Gradient Boosting Regressor</div><div class="badge badge-green">Production Ready</div></div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: DEMAND DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="demand-dynamics" class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Temporal Demand Dynamics</div><div class="section-sub">Understand how seasonality and special events dictate purchasing volume. Observe the velocity of sales over months and the distinct uplift generated by holiday periods.</div>', unsafe_allow_html=True)

monthly_df = df.groupby('month')['qty'].agg(['sum', 'mean']).reset_index()
monthly_df.columns = ['MonthNum', 'Total Demand', 'Avg Order Size']
monthly_df['Month'] = [MONTHS[i-1] for i in monthly_df['MonthNum']]

c1, c2 = st.columns([1.2, 1], gap="large")
with c1:
    with st.container(border=True):
        st.markdown('<div class="card-title">Seasonal Velocity</div><div class="card-desc">Interactive dual-axis tracking: Total monthly volume (bars) vs. Average units per order (line curve)</div>', unsafe_allow_html=True)
        fig_season = make_subplots(specs=[[{"secondary_y": True}]])
        fig_season.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['Total Demand'], name="Total Volume", marker_color=C_BLU, opacity=0.9), secondary_y=False)
        fig_season.add_trace(go.Scatter(x=monthly_df['Month'], y=monthly_df['Avg Order Size'], name="Avg Order Size", mode="lines+markers", line=dict(color=C_TEA, width=4, shape="spline"), marker=dict(size=10, color=C_TEA, line=dict(width=2, color="white"))), secondary_y=True)
        st.plotly_chart(apply_plotly_theme(fig_season), use_container_width=True, height=400)

with c2:
    with st.container(border=True):
        st.markdown('<div class="card-title">Year-Over-Year Trajectory</div><div class="card-desc">Comparing monthly demand momentum across 2017 and 2018</div>', unsafe_allow_html=True)
        if df['year'].nunique() > 1:
            yoy_df = df.groupby(['year', 'month'])['qty'].sum().reset_index()
            yoy_df['year'] = yoy_df['year'].astype(str)
            fig_yoy = px.line(yoy_df, x='month', y='qty', color='year', markers=True, color_discrete_sequence=[C_PUR, C_AMB])
            fig_yoy.update_traces(line=dict(width=4, shape='spline'), marker=dict(size=10, line=dict(width=2, color="white")))
            fig_yoy.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTHS)
            st.plotly_chart(apply_plotly_theme(fig_yoy), use_container_width=True, height=400)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: PRICE ELASTICITY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="price-elasticity" class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Price Elasticity & Market Intelligence</div><div class="section-sub">Dive into how pricing strategies and shipping costs affect total volume. The dataset benchmarks our internal pricing directly against three major market competitors.</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(border=True):
        st.markdown('<div class="card-title">Elasticity Scatter Matrix</div><div class="card-desc">Individual transactions plotted against an OLS regression trendline. A negative correlation confirms elastic demand.</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(df, x="unit_price", y="qty", opacity=0.6, color_discrete_sequence=[C_BLU], trendline="ols", trendline_color_override=C_COR)
        fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        st.plotly_chart(apply_plotly_theme(fig_scatter), use_container_width=True, height=400)

with c2:
    with st.container(border=True):
        st.markdown('<div class="card-title">Competitor Pricing Distribution</div><div class="card-desc">Box plot analysis comparing our pricing strategy against tracked market competitors.</div>', unsafe_allow_html=True)
        comp_cols = [c for c in ['comp_1', 'comp_2', 'comp_3'] if c in df.columns]
        if len(comp_cols) > 0:
            box_data = [df['unit_price']] + [df[c] for c in comp_cols]
            labels = ['Our Price', 'Comp 1', 'Comp 2', 'Comp 3']
            fig_box = go.Figure()
            for idx, data in enumerate(box_data):
                fig_box.add_trace(go.Box(y=data, name=labels[idx], marker_color=[C_BLU, C_PUR, C_TEA, C_AMB][idx], boxmean=True, boxpoints='outliers', line_width=3))
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(apply_plotly_theme(fig_box), use_container_width=True, height=400)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: AI PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="ai-predictor" class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">AI Forecast Engine</div><div class="section-sub">Simulate live market conditions. Adjust pricing, projected traffic, and seasonality parameters to instantly run the Gradient Boosting model and predict order volume.</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1.1, 1], gap="large")

with c1:
    with st.form("predict_form", border=True):
        st.markdown('<div class="card-title">Model Parameters</div><div class="card-desc" style="margin-bottom:1rem">Tune inputs to generate a live prediction</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2, gap="medium")
        with f1:
            customers = st.number_input("Customer Traffic (Est.)", value=50, step=5)
            unit_price = st.number_input("Target Unit Price ($)", value=75.0, step=5.0)
            freight_price = st.number_input("Estimated Freight ($)", value=15.0, step=1.0)
            product_score = st.number_input("Product Rating (1-5)", value=4.5, step=0.1)
        with f2:
            month = st.selectbox("Sales Month", range(1, 13), index=5, format_func=lambda x: MONTHS[x-1])
            year = st.selectbox("Sales Year", [2017, 2018], index=1)
            holiday = st.selectbox("Holiday Period?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            stock = st.number_input("Current Inventory", value=40, step=5)
        submitted = st.form_submit_button("Launch AI Prediction")

with c2:
    with st.container(border=True):
        st.markdown('<div class="card-title">Forecast Output</div><div class="card-desc">Real-time model inference results</div>', unsafe_allow_html=True)
        if submitted:
            input_df = pd.DataFrame(0, index=[0], columns=df.drop("qty", axis=1).columns)
            input_df['customers'] = customers; input_df['unit_price'] = unit_price
            input_df['freight_price'] = freight_price; input_df['product_score'] = product_score
            input_df['month'] = month; input_df['year'] = year; input_df['holiday'] = holiday
            
            try:
                pred_qty = max(0, round(model.predict(input_df)[0]))
                alert_html = ""
                if pred_qty > stock:
                    alert_html = f'<div style="background:var(--coral-lt);color:var(--coral);border:2px solid var(--coral);font-size:16px;padding:16px;border-radius:12px;font-weight:700;">⚠️ Restock required: Demand exceeds inventory by {pred_qty-stock} units.</div>'
                elif stock > pred_qty * 2.5:
                    alert_html = f'<div style="background:var(--amber-lt);color:var(--amber);border:2px solid var(--amber);font-size:16px;padding:16px;border-radius:12px;font-weight:700;">📦 Overstock risk: Inventory is highly saturated vs. forecast.</div>'
                else:
                    alert_html = f'<div style="background:var(--teal-lt);color:var(--teal);border:2px solid var(--teal);font-size:16px;padding:16px;border-radius:12px;font-weight:700;">✅ Inventory balanced: Current stock aligns perfectly with demand.</div>'

                st.markdown(f"""
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:350px;text-align:center;gap:12px">
                    <div style="font-size:18px;color:var(--text-sub);text-transform:uppercase;letter-spacing:0.15em;font-weight:800">Predicted Demand</div>
                    <div style="font-family:'Outfit';font-size:9rem;font-weight:900;background:linear-gradient(135deg,var(--blue),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1; filter: drop-shadow(0px 15px 25px rgba(37,99,235,0.25));">{pred_qty}</div>
                    <div style="font-size:19px;color:var(--text-main);font-weight:700;margin-bottom:1.5rem">units expected to sell</div>
                    <div style="width:100%;max-width:400px;">{alert_html}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;min-height:350px;background:#f8fafc;border-radius:16px;border:3px dashed #cbd5e1;">
                <p style="font-size:18px;color:var(--text-sub);line-height:1.7;max-width:300px;text-align:center;font-weight:600;">Tune the parameters on the left and click <b style="color:var(--blue)">Launch AI Prediction</b> to run inference.</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)