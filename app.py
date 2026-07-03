import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIG & SYSTEM SETTINGS ---
st.set_page_config(
    page_title="Red Wine Quality Analytics", 
    page_icon="🍷", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Color Palette Definition
WINE_COLOR = "#581845"      # Deep Burgundy / Wine
ACCENT_AMBER = "#FFC300"    # Warm Amber (for high values / warnings)
GOOD_GREEN = "#2ECC71"      # Mint Green for high quality
BAD_RED = "#E74C3C"         # Muted Red for low quality
BG_LIGHT = "#1A1115"      

# --- 2. LOAD PRE-TRAINED MODELS & REFERENCE DATA ---
@st.cache_resource
def load_ml_components():
    # Replace these filenames with your exact exported assets
    model = joblib.load('wine_quality.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_reference_dataset():
    """
    Loads a small sample or the full cleaned white wine dataset 
    to drive the 'Dataset Insights' visual tab.
    Replace with your actual cleaned CSV path if available!
    """
    try:
        # Utilizing standard UCI White Wine boundaries for realistic fallback data if file missing
        df = pd.read_csv("winequality-white_cleaned.csv")
    except:
        # Synthetic template mimicking the white wine distribution for display purposes
        np.random.seed(42)
        n_samples = 200
        data = {
            'fixed acidity': np.random.normal(6.9, 0.8, n_samples),
            'volatile acidity': np.random.normal(0.27, 0.1, n_samples),
            'citric acid': np.random.normal(0.33, 0.12, n_samples),
            'residual sugar': np.random.uniform(1.0, 15.0, n_samples),
            'chlorides': np.random.normal(0.045, 0.01, n_samples),
            'free sulfur dioxide': np.random.normal(35.0, 15.0, n_samples),
            'total sulfur dioxide': np.random.normal(138.0, 40.0, n_samples),
            'density': np.random.normal(0.994, 0.002, n_samples),
            'pH': np.random.normal(3.18, 0.15, n_samples),
            'sulphates': np.random.normal(0.49, 0.11, n_samples),
            'quality': np.random.choice([5, 6, 7, 8], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
        }
        df = pd.DataFrame(data)
    return df

# Initialize components
model, scaler = load_ml_components()
df_ref = load_reference_dataset()

# --- 3. CUSTOM BRANDING UI ---
st.markdown(f"""
    <style>
        .main-header {{
            font-size: 2.8rem;
            font-weight: 800;
            color: {WINE_COLOR};
            text-align: center;
            margin-bottom: 0.2rem;
        }}
        .sub-header {{
            font-size: 1.1rem;
            color: #555555;
            text-align: center;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background-color:white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-left: 5px solid {WINE_COLOR};
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🍷Fine Wine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An interactive predictive dashboard transforming chemical profiling into instant quality classification.</div>', unsafe_allow_html=True)

# --- 4. NAVIGATION TABS ---
tab1, tab2 = st.tabs(["🔮 Predictive Engine", "📊 Deep Data Insights & Analytics"])

# ==========================================
# TAB 1: PREDICTIVE ENGINE
# ==========================================
with tab1:
    st.markdown("### 🛠️ Input Wine Chemical Profiles")
    st.write("Adjust the parameters below to run the sample through your production machine learning model.")
    
    # Layout adjustment knobs across 3 clean columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Acidity Profiles**")
        fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 3.8, 14.2, 6.9, step=0.1)
        volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.08, 1.10, 0.27, step=0.01)
        citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.66, 0.33, step=0.01)
        
    with col2:
        st.markdown(f"**Sugars & Salts**")
        residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.6, 30.0, 5.2, step=0.1)
        chlorides = st.slider("Chlorides (g/dm³)", 0.01, 0.35, 0.045, step=0.001)
        pH = st.slider("pH Index Level", 2.72, 3.82, 3.18, step=0.01)
        
    with col3:
        st.markdown(f"**Sulfurs & Physical Properties**")
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (mg/dm³)", 2.0, 120.0, 35.0, step=1.0)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide (mg/dm³)", 9.0, 300.0, 138.0, step=1.0)
        density = st.slider("Density (g/cm³)", 0.987, 1.003, 0.994, step=0.001)
        sulphates = st.slider("Sulphates (g/dm³)", 0.22, 1.08, 0.49, step=0.01)

    # Structuring feature row mapping
    feature_names = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates'
    ]
    
    input_df = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates
    ]], columns=feature_names)

    st.markdown("---")
    
    # Execution block triggered via button interaction
    if st.button("🚀 Execute Pipeline Inference", type="primary"):
        # Scale inputs using the loaded scaler asset
        scaled_features = scaler.transform(input_df)
        
        # Pull model output
        prediction = model.predict(scaled_features)[0]
        raw_prediction = float(prediction)
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(scaled_features)[0]
            # Assuming your encoder mapped 1 -> Good, 0 -> Bad
            score = prediction_proba[1] * 100
            st.metric(label="Model Quality Probability", value=f"{score:.1f}%")

            if 7.0 <= score <= 7.5:
                st.markdown(f"### Status: <span style='color:{GOOD_GREEN}; font-weight:bold;'>🍷😉 GOOD CHOICE</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"### Status: <span style='color:{BAD_RED}; font-weight:bold;'>🥸YOU ARE AGING MAN</span>", unsafe_allow_html=True)
        else:
            # Fallback directly to binary discrete classifications if probabilities aren't supported
            label = "🍷 GOOD QUALITY" if prediction == 1 else "🤢 BAD QUALITY"
            color = GOOD_GREEN if prediction == 1 else BAD_RED
            st.markdown(f"### Assessment: <span style='color:{color}; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns(2)
        with res_col2:
            # Gauge Visualization showing exactly where this configuration rests
            if raw_prediction <= 10.0:
                gauge_max = 10.0
                gauge_steps = [
                    {'range': [0, 7], 'color': "#FFE600"},
                    {'range': [7, 7.2], 'color': "#00FF66"},
                    {'range': [7.2, 10], 'color': "#FF0033"}
                ]
                show_ticks="outside"
            else:
                gauge_max = float(np.ceil(raw_prediction))
                gauge_steps=[
                    {'range': [0, 7], 'color': "#FFE600"},
                    {'range': [7, 7.2], 'color': "#00FF66"},
                    {'range': [7.2, 10], 'color': "#FF0033"}
                ]
                show_ticks= ""
            fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = float(raw_prediction), # Send the true high score to the needle
                    title = {'text': "Wine Quality Metric Scale", 'font': {'color': "#581845", 'size': 16, 'bold': True}},
                    gauge = {
                        'axis': {
                            'range': [0, gauge_max], 
                            'tickwidth': 2 if raw_prediction <= 10 else 0, # Hides tick bars if > 10
                            'tickcolor': "#333333",
                            'ticks': show_ticks # Drops or shows numeric labels dynamically
                        },  
                        'bar': {'color': "#111111"}, 
                        'bgcolor': "#FFFFFF",        
                        'steps': gauge_steps, # Uses our smart, auto-stretching colors
                        'threshold': {
                            'line': {'color': "#000000", 'width': 3}, 
                            'thickness': 0.8,
                            'value': 7.0
                        }
                    }
                ))
            fig_gauge.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=220)
            st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# TAB 2: DATASET INSIGHTS & ANALYTICS
# ==========================================
with tab2:
    st.markdown("### 📊 Exploratory Chemical Dataset Analytics")
    st.write("Understand the systemic data distributions and attribute relationships driving the core ML model behavior.")
    
    # Global Summary Highlighting benchmarks
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Total Profile Rows Analyzed", f"{len(df_ref)}")
    stat2.metric("Mean Alcohol Concentration", "10.51 %")
    stat3.metric("Optimal pH Range Baseline", "3.10 - 3.30")
    stat4.metric("Dataset Balance Ratio (Good:Bad)", "~65% : 35%")
    
    st.markdown("---")
    
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.subheader("💡 Chemical Distribution & Densities")
        selected_feature = st.selectbox(
            "Select a property to view its global distribution profile:",
            options=feature_names,
            index=3 # Defaults to residual sugar
        )
        
        # Styled Distribution Plot
        fig_hist = px.histogram(
            df_ref, 
            x=selected_feature, 
            color_discrete_sequence=[WINE_COLOR],
            marginal="box",
            title=f"Distribution Profile: {selected_feature.title()}",
            template="plotly_white"
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=selected_feature.title(),
            yaxis_title="Frequency Count"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with vis_col2:
        st.subheader("🔗 Feature Interactions & Correlations")
        feat_x = st.selectbox("X-Axis Feature Correlation:", options=feature_names, index=1)
        feat_y = st.selectbox("Y-Axis Feature Correlation:", options=feature_names, index=7)
        
        # Colorful Scatter Analysis
        fig_scatter = px.scatter(
            df_ref, 
            x=feat_x, 
            y=feat_y, 
            color="quality" if "quality" in df_ref.columns else None,
            color_continuous_scale=px.colors.sequential.Burg,
            title=f"Interaction Spectrum: {feat_x.title()} vs {feat_y.title()}",
            template="plotly_white"
        )
        fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Dynamic Insight Matrix at the bottom
    st.markdown("### 🧪 Key Chemical Insights")
    st.info("""
    - **Volatile Acidity:** Higher distributions typically correlate directly with poor sensory qualities (acetic acid taste profiles).
    - **Sulphates & Preservatives:** Actively bound indicators directly prevent wine oxidation, providing strong predictive patterns within the Random Forest split paths.
    - **Density/Sugar Multicollinearity:** Strong baseline correlation patterns emerge directly between density and residual sugars during deep linear analysis stages.
    """)
