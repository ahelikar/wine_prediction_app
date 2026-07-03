import streamlit as st

# Define your palette variables
PAGE_BACKGROUND = "#F4F6F7"   # Replace with your desired background HEX code
TEXT_COLOR = "#2C3E50"        # Replace with your text color HEX code

# Inject structural layout overrides
st.markdown(f"""
    <style>
        /* Target all of Streamlit's main content wrapper blocks */
        .stApp, 
        .stAppViewContainer, 
        .stMainBlockContainer, 
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {{
            background-color: {PAGE_BACKGROUND} !important;
            background: {PAGE_BACKGROUND} !important;
        }}
        
        /* Force text colors to match your theme across text blocks and headers */
        .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {{
            color: {TEXT_COLOR} !important;
        }}
    </style>
""", unsafe_allowed_html=True)
