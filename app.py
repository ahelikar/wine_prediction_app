import streamlit as st
import joblib
import plotly.express as px
import pandas as pd
import numpy as np
def load_model():
    model = joblib.load('wine_quality.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.set_page_config(page_title="White Wine Quality Predictor", page_icon="🍷", layout="wide")
st.title("🍷 White Wine Quality Portfolio Dashboard")
st.markdown("""
This application utilizes a Machine Learning model trained on the White Wine Quality dataset to classify wine variants as **Good** or **Bad** based on their chemical composition. 
""")

# Setup tabs to separate predictions from documentation/insights
tab1, tab2 = st.tabs(["🔮 Interactive Predictor", "📊 Dataset Insights & Project Specs"])

with tab1:
    st.subheader("Adjust Wine Chemical Properties")
    
    # Split input controls into 3 balanced columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 3.8, 14.2, 7.0, step=0.1)
        volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.10, 0.27, step=0.01)
        citric_acid = st.slider("Citric Acid", 0.0, 1.66, 0.34, step=0.01)
        
    with col2:
        residual_sugar = st.slider("Residual Sugar", 0.6, 30.0, 5.2, step=0.1)
        chlorides = st.slider("Chlorides", 0.01, 0.35, 0.04, step=0.001)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 2.0, 120.0, 35.0, step=1.0)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 9.0, 300.0, 138.0, step=1.0)
        
    with col3:
        density = st.slider("Density", 0.987, 1.003, 0.994, step=0.001)
        pH = st.slider("pH Level", 2.72, 3.82, 3.18, step=0.01)
        sulphates = st.slider("Sulphates", 0.22, 1.08, 0.49, step=0.01)

    # Reconstruct the feature array exactly how your model expects it
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

    # 3. Execution Pipeline
    if st.button("Analyze Wine Quality", type="primary"):
        # Scale inputs using the loaded scaler
        scaled_features = scaler.transform(input_df)
        
        # Predict Class and Probabilities
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        # Interpret LabelEncoder mappings (Assuming 1 is Good, 0 is Bad)
        # Adjust strings if your notebook encoded them differently
        result_label = "🍷 GOOD QUALITY" if prediction == 1 else "🤢 BAD QUALITY"
        result_color = "green" if prediction == 1 else "red"
        
        st.markdown(f"### Result: :{result_color}[{result_label}]")
        
        # Plot Confidence Probability with Plotly
        prob_df = pd.DataFrame({
            "Category": ["Bad Quality", "Good Quality"],
            "Confidence (%)": prediction_proba * 100
        })
        
        fig = px.bar(prob_df, x="Confidence (%)", y="Category", orientation='h',
                     title="Model Prediction Confidence",
                     color="Category", color_discrete_map={"Bad Quality": "#ef553b", "Good Quality": "#00cc96"})
        fig.update_layout(xaxis_max=100)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Project Overview & Model Architecture")
    st.markdown("""
    ### Technical Framework
    - **Data Pipeline:** Cleaned missing entries, eliminated duplicated rows, and addressed skewness.
    - **Preprocessing:** All scalar inputs are passed through a `StandardScaler` to align with unit variance bounds before evaluation.
    - **Models Tested:** Logistic Regression, Random Forest, Support Vector Machines, and Gradient Boosting Classifiers.
    """)