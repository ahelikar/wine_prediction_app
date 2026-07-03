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
    # 3. Execution Pipeline
    if st.button("Analyze Wine Quality", type="primary"):
        # Scale inputs using the loaded scaler
        scaled_features = scaler.transform(input_df)
        
        # 1. Get the predicted quality rating (e.g., 5.4, 6.1, etc.)
        predicted_score = model.predict(scaled_features)[0]
        
        st.subheader("🔮 Prediction Results")
        
        # Display the exact predicted score nicely
        st.metric(label="Predicted Quality Score", value=f"{predicted_score:.2f} / 10")
        
        # 2. Determine if it's Good or Bad based on a threshold (e.g., 6.0 and above is Good)
        # Adjust this threshold number to match what you used in your notebook!
        if predicted_score >= 6.0:
            result_label = "🍷 GOOD QUALITY"
            result_color = "green"
            description = "This wine has excellent balance and desirable chemical properties."
        else:
            result_label = "🤢 BAD QUALITY"
            result_color = "red"
            description = "This wine falls below the standard quality benchmark."
            
        st.markdown(f"### Assessment: :{result_color}[{result_label}]")
        st.info(description)
        
        # 3. Optional: Visualizing where this score sits on a 0-10 gauge
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Quality Level"},
            gauge = {
                'axis': {'range': [0, 10]},
                'bar': {'color': "#6A1B29"}, # Wine color
                'steps': [
                    {'range': [0, 6], 'color': "#F5F5F5"},
                    {'range': [6, 10], 'color': "#E8F5E9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 6.0
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)