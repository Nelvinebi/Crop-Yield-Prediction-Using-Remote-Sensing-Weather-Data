"""
🌾 Crop Yield Prediction Dashboard
Author: AGBOZU EBINGIYE NELVIN
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME & GLOBAL CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --soil:   #2C1A0E;
    --earth:  #4A2C17;
    --amber:  #D4862A;
    --gold:   #F0B941;
    --straw:  #F7DFA0;
    --leaf:   #4A7C59;
    --sage:   #7BAE8C;
    --mist:   #EBF4EE;
    --sky:    #1B3A5C;
    --dusk:   #2D5F8A;
    --cream:  #FEFAF4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream);
    color: var(--soil);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--soil) 0%, var(--earth) 100%);
    border-right: 3px solid var(--amber);
}
section[data-testid="stSidebar"] * {
    color: var(--straw) !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: var(--gold) !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-size: 0.78rem;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--amber) !important;
    opacity: 0.3;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, var(--soil) 0%, var(--earth) 40%, var(--leaf) 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🌾🌿🌱";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 4rem;
    opacity: 0.15;
    letter-spacing: 0.5rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: var(--straw) !important;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero p {
    color: var(--sage) !important;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.02em;
}
.hero .badge {
    display: inline-block;
    background: var(--amber);
    color: var(--soil) !important;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border-left: 5px solid var(--amber);
    box-shadow: 0 4px 20px rgba(44,26,14,0.07);
    position: relative;
}
.metric-card.green  { border-left-color: var(--leaf); }
.metric-card.sky    { border-left-color: var(--sky); }
.metric-card.gold   { border-left-color: var(--gold); }
.metric-card .icon  { font-size: 1.6rem; margin-bottom: 0.4rem; }
.metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    font-weight: 600;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--soil);
    line-height: 1.1;
}
.metric-card .sub { font-size: 0.78rem; color: #aaa; margin-top: 0.2rem; }

/* Section headers */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--earth);
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, var(--leaf), var(--sky));
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white !important;
    box-shadow: 0 8px 30px rgba(74,124,89,0.3);
}
.pred-box .pred-value {
    font-family: 'DM Serif Display', serif;
    font-size: 4rem;
    line-height: 1;
    color: var(--gold);
}
.pred-box .pred-label { font-size: 1rem; opacity: 0.85; }

/* Tab styling */
.stTabs [role="tablist"] { gap: 0.5rem; }
.stTabs [role="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.5rem 1.2rem;
    border-radius: 8px 8px 0 0;
    background: rgba(74,124,89,0.08);
    color: var(--earth) !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: var(--leaf);
    color: white !important;
}

/* Plotly chart containers */
.chart-wrap {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 4px 20px rgba(44,26,14,0.06);
    margin-bottom: 1rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: #bbb;
    font-size: 0.78rem;
    border-top: 1px solid #eee;
    margin-top: 3rem;
}
.footer a { color: var(--leaf); text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD & CACHE DATA ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("data/data.xlsx")
    return df

@st.cache_resource
def train_model(test_size, n_estimators):
    df = load_data()
    X = df.drop("Crop_Yield_t_ha", axis=1)
    y = df["Crop_Yield_t_ha"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, X_test, y_test, y_pred, rmse, r2, feat_imp, X.columns.tolist()

data = load_data()

FEATURES = ["NDVI", "EVI", "Rainfall_mm", "Temperature_C", "Humidity_%", "Soil_Moisture"]
FEAT_LABELS = {
    "NDVI": "NDVI",
    "EVI": "EVI",
    "Rainfall_mm": "Rainfall (mm)",
    "Temperature_C": "Temperature (°C)",
    "Humidity_%": "Humidity (%)",
    "Soil_Moisture": "Soil Moisture",
}

COLORS = {
    "amber": "#D4862A",
    "gold":  "#F0B941",
    "leaf":  "#4A7C59",
    "sage":  "#7BAE8C",
    "sky":   "#1B3A5C",
    "dusk":  "#2D5F8A",
    "soil":  "#2C1A0E",
    "earth": "#4A2C17",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="white",
    plot_bgcolor="#FEFAF4",
    font=dict(family="DM Sans", color="#2C1A0E"),
)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌾 Model Settings")
    st.markdown("---")
    n_estimators = st.slider("Trees in Forest", 50, 300, 100, 10)
    test_size    = st.slider("Test Split (%)", 10, 40, 20, 5) / 100
    st.markdown("---")
    st.markdown("### 🎯 Yield Predictor")
    st.caption("Adjust environmental inputs:")
    ndvi      = st.slider("NDVI",            float(data.NDVI.min()),       float(data.NDVI.max()),       float(data.NDVI.mean()),       0.01)
    evi       = st.slider("EVI",             float(data.EVI.min()),        float(data.EVI.max()),        float(data.EVI.mean()),        0.01)
    rainfall  = st.slider("Rainfall (mm)",   float(data.Rainfall_mm.min()),float(data.Rainfall_mm.max()),float(data.Rainfall_mm.mean()),10.0)
    temp      = st.slider("Temperature (°C)",float(data.Temperature_C.min()),float(data.Temperature_C.max()),float(data.Temperature_C.mean()),0.5)
    humidity  = st.slider("Humidity (%)",    float(data["Humidity_%"].min()),float(data["Humidity_%"].max()),float(data["Humidity_%"].mean()),1.0)
    soil      = st.slider("Soil Moisture",   float(data.Soil_Moisture.min()),float(data.Soil_Moisture.max()),float(data.Soil_Moisture.mean()),0.01)
    st.markdown("---")
    st.markdown("<small>Built by **Agbozu Nelvin**<br>Environmental Data Scientist</small>", unsafe_allow_html=True)

# ─── TRAIN MODEL ────────────────────────────────────────────────────────────
model, X_test, y_test, y_pred, rmse, r2, feat_imp, feature_cols = train_model(test_size, n_estimators)

user_input = np.array([[ndvi, evi, rainfall, temp, humidity, soil]])
user_prediction = model.predict(user_input)[0]

# ─── HERO ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">Remote Sensing · Machine Learning · Agriculture</div>
    <h1>🌾 Crop Yield Intelligence</h1>
    <p>Integrating NDVI, EVI, weather & soil data with Random Forest for precision yield forecasting</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI METRICS ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card gold">
        <div class="icon">📊</div>
        <div class="label">Dataset Size</div>
        <div class="value">{len(data)}</div>
        <div class="sub">samples · 6 features</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card green">
        <div class="icon">🎯</div>
        <div class="label">R² Score</div>
        <div class="value">{r2:.3f}</div>
        <div class="sub">variance explained</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="icon">📉</div>
        <div class="label">RMSE</div>
        <div class="value">{rmse:.3f}</div>
        <div class="sub">t/ha prediction error</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card sky">
        <div class="icon">🌿</div>
        <div class="label">Avg Yield</div>
        <div class="value">{data.Crop_Yield_t_ha.mean():.2f}</div>
        <div class="sub">tons per hectare</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── YIELD PREDICTOR ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🎯 Live Yield Predictor</div>', unsafe_allow_html=True)
pred_col, gauge_col = st.columns([1, 1])

with pred_col:
    category = "Low" if user_prediction < 4 else ("High" if user_prediction > 9 else "Moderate")
    color    = "#D4862A" if category == "Low" else ("#4A7C59" if category == "High" else "#2D5F8A")
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-label">Predicted Crop Yield</div>
        <div class="pred-value">{user_prediction:.2f}</div>
        <div class="pred-label">tons per hectare (t/ha)</div>
        <br>
        <span style="background:rgba(255,255,255,0.2);padding:0.3rem 1rem;border-radius:20px;font-size:0.85rem;font-weight:600;">
            {category} Yield Zone
        </span>
    </div>
    """, unsafe_allow_html=True)

with gauge_col:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=user_prediction,
        delta={"reference": data.Crop_Yield_t_ha.mean(), "valueformat": ".2f"},
        title={"text": "vs. Dataset Average", "font": {"size": 13, "family": "DM Sans"}},
        number={"suffix": " t/ha", "font": {"size": 28, "family": "DM Serif Display"}},
        gauge={
            "axis": {"range": [data.Crop_Yield_t_ha.min(), data.Crop_Yield_t_ha.max()],
                     "tickfont": {"size": 10}},
            "bar":  {"color": COLORS["leaf"]},
            "steps": [
                {"range": [data.Crop_Yield_t_ha.min(), 4],  "color": "#FDEBD0"},
                {"range": [4, 9],                            "color": "#D5F5E3"},
                {"range": [9, data.Crop_Yield_t_ha.max()],  "color": "#AED6F1"},
            ],
            "threshold": {"line": {"color": COLORS["amber"], "width": 3},
                          "thickness": 0.8,
                          "value": data.Crop_Yield_t_ha.mean()},
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20), **PLOTLY_TEMPLATE)
    st.plotly_chart(fig_gauge, use_container_width=True)


# ─── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance",
    "🔬 Feature Analysis",
    "🗺️ Data Explorer",
    "📋 Raw Data",
])

# ── TAB 1: MODEL PERFORMANCE ────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Model Validation</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # Actual vs Predicted
        residuals = np.array(y_test) - np.array(y_pred)
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=list(y_test), y=list(y_pred),
            mode="markers",
            marker=dict(color=residuals, colorscale="RdYlGn",
                        size=8, opacity=0.8,
                        colorbar=dict(title="Residual", thickness=12),
                        showscale=True),
            name="Predictions",
            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        ))
        lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
        fig_avp.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
            line=dict(color=COLORS["amber"], dash="dash", width=2), name="Perfect Fit"))
        fig_avp.update_layout(
            title="Actual vs Predicted Yield",
            xaxis_title="Actual Yield (t/ha)",
            yaxis_title="Predicted Yield (t/ha)",
            height=380, **PLOTLY_TEMPLATE,
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    with c2:
        # Residual distribution
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=20,
            marker_color=COLORS["leaf"], opacity=0.8,
            name="Residuals",
        ))
        fig_res.add_vline(x=0, line_dash="dash", line_color=COLORS["amber"], line_width=2)
        fig_res.update_layout(
            title="Residual Distribution",
            xaxis_title="Residual (t/ha)",
            yaxis_title="Count",
            height=380, **PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # Prediction vs actual line over sorted samples
    sorted_idx = np.argsort(np.array(y_test))
    sorted_actual = np.array(y_test)[sorted_idx]
    sorted_pred   = np.array(y_pred)[sorted_idx]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=list(range(len(sorted_actual))), y=sorted_actual,
        mode="lines+markers", name="Actual",
        line=dict(color=COLORS["earth"], width=2),
        marker=dict(size=4)))
    fig_line.add_trace(go.Scatter(x=list(range(len(sorted_pred))), y=sorted_pred,
        mode="lines", name="Predicted",
        line=dict(color=COLORS["leaf"], width=2, dash="dot")))
    fig_line.update_layout(
        title="Actual vs Predicted — Sorted Test Samples",
        xaxis_title="Sample Index (sorted by actual yield)",
        yaxis_title="Yield (t/ha)",
        height=320, **PLOTLY_TEMPLATE,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_line, use_container_width=True)


# ── TAB 2: FEATURE ANALYSIS ─────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Feature Importance & Correlation</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)

    with f1:
        # Feature importance horizontal bar
        fi_df = feat_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Label"] = fi_df["Feature"].map(FEAT_LABELS)
        fig_fi = px.bar(fi_df.sort_values("Importance"),
                        x="Importance", y="Label", orientation="h",
                        color="Importance", color_continuous_scale=["#D5F5E3", COLORS["leaf"], COLORS["sky"]],
                        text=fi_df.sort_values("Importance")["Importance"].apply(lambda v: f"{v:.3f}"))
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            title="Feature Importance (Random Forest)",
            xaxis_title="Importance Score",
            yaxis_title="",
            coloraxis_showscale=False,
            height=380, **PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with f2:
        # Correlation heatmap
        corr = data.rename(columns=FEAT_LABELS | {"Crop_Yield_t_ha": "Crop Yield"}).corr()
        fig_heat = px.imshow(corr, text_auto=".2f",
                             color_continuous_scale=["#4A2C17", "#FEFAF4", "#4A7C59"],
                             zmin=-1, zmax=1, aspect="auto")
        fig_heat.update_layout(
            title="Feature Correlation Matrix",
            height=380, **PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Scatter matrix of top 4 features
    st.markdown('<div class="section-title">Scatter — Feature vs Yield</div>', unsafe_allow_html=True)
    sel_feat = st.selectbox("Select feature to plot against Yield",
                             [FEAT_LABELS[f] for f in FEATURES],
                             index=0)
    raw_feat = {v: k for k, v in FEAT_LABELS.items()}[sel_feat]

    fig_sc = px.scatter(data, x=raw_feat, y="Crop_Yield_t_ha",
                        color="Crop_Yield_t_ha",
                        color_continuous_scale=["#D4862A", "#F0B941", "#4A7C59"],
                        trendline="lowess",
                        labels={raw_feat: sel_feat, "Crop_Yield_t_ha": "Crop Yield (t/ha)"},
                        opacity=0.75)
    fig_sc.update_layout(
        title=f"{sel_feat} vs Crop Yield",
        height=360, **PLOTLY_TEMPLATE,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ── TAB 3: DATA EXPLORER ────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Dataset Distribution Explorer</div>', unsafe_allow_html=True)

    # Distribution of each feature
    fig_dist = make_subplots(rows=2, cols=3,
        subplot_titles=[FEAT_LABELS[f] for f in FEATURES])
    palette = [COLORS["leaf"], COLORS["dusk"], COLORS["amber"],
               COLORS["earth"], COLORS["sage"], COLORS["sky"]]
    for i, feat in enumerate(FEATURES):
        r, c = divmod(i, 3)
        fig_dist.add_trace(
            go.Histogram(x=data[feat], marker_color=palette[i], opacity=0.8,
                         showlegend=False, nbinsx=20, name=FEAT_LABELS[feat]),
            row=r+1, col=c+1
        )
    fig_dist.update_layout(height=420, **PLOTLY_TEMPLATE,
                            title_text="Feature Distributions",
                            margin=dict(t=60))
    st.plotly_chart(fig_dist, use_container_width=True)

    # Yield distribution
    e1, e2 = st.columns(2)
    with e1:
        fig_yield = go.Figure()
        fig_yield.add_trace(go.Histogram(x=data["Crop_Yield_t_ha"],
            marker_color=COLORS["leaf"], opacity=0.85, nbinsx=25))
        fig_yield.add_vline(x=data.Crop_Yield_t_ha.mean(),
            line_dash="dash", line_color=COLORS["amber"], line_width=2,
            annotation_text="Mean", annotation_position="top right")
        fig_yield.update_layout(
            title="Crop Yield Distribution",
            xaxis_title="Yield (t/ha)", yaxis_title="Count",
            height=320, **PLOTLY_TEMPLATE)
        st.plotly_chart(fig_yield, use_container_width=True)

    with e2:
        # Box plots per feature (normalized)
        norm = (data[FEATURES] - data[FEATURES].min()) / (data[FEATURES].max() - data[FEATURES].min())
        norm.columns = [FEAT_LABELS[f] for f in FEATURES]
        fig_box = go.Figure()
        for i, col in enumerate(norm.columns):
            fig_box.add_trace(go.Box(y=norm[col], name=col,
                marker_color=palette[i], line_color=palette[i], showlegend=False))
        fig_box.update_layout(
            title="Normalised Feature Distributions",
            yaxis_title="Normalised Value",
            height=320, **PLOTLY_TEMPLATE)
        st.plotly_chart(fig_box, use_container_width=True)


# ── TAB 4: RAW DATA ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    display = data.rename(columns=FEAT_LABELS | {"Crop_Yield_t_ha": "Crop Yield (t/ha)"})
    st.dataframe(display.style
        .background_gradient(subset=["Crop Yield (t/ha)"], cmap="YlGn")
        .format(precision=3),
        use_container_width=True, height=400)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Summary Statistics**")
        st.dataframe(display.describe().round(3), use_container_width=True)
    with d2:
        st.markdown("**Missing Values**")
        miss = display.isnull().sum().reset_index()
        miss.columns = ["Feature", "Missing"]
        st.dataframe(miss, use_container_width=True)
    with d3:
        st.markdown("**Download Data**")
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv, "crop_yield_data.csv", "text/csv")


# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🌾 Crop Yield Intelligence Dashboard &nbsp;·&nbsp;
    Built by <strong>Agbozu Ebingiye Nelvin</strong> — Environmental Data Scientist &nbsp;·&nbsp;
    <a href="https://github.com/Nelvinebi" target="_blank">GitHub</a> &nbsp;·&nbsp;
    <a href="https://www.linkedin.com/in/agbozu-ebi/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
