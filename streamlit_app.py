import streamlit as st
import pandas as pd
from graphviz import Digraph
import pickle
import requests
from io import BytesIO
import bz2

st.title('🚀 Cuasal ML')

st.info('By leveraging counterfactual causal reasoning and advanced machine learning techniques, we seek to provide deep insights into the dynamics of the digital freelance economy.')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Economist-Ahmed-Dawoud/Data/main/streamlit-upwork-data.csv')
  df
  
with st.expander('DAG'):
  def create_freelance_dag():
    # Graph Viz DAG
    dot = Digraph(comment='Freelancer Success DAG')
    dot.attr(rankdir='TB')
    
    
    # Node definitions
    nodes = [
        "Full-Stack", "Front-End", "Back-End", "Other-Development", "Other-Tech",
        "Creative-Design", "Business-Marketing", "Other-Occupation", "portfolio",
        "certifs", "video", "testims", "jobs", "hour_rate_clean", "success_rate",
        "rate_cat", "hours_per_week", "hours", "earnings", "num-occups", "country",
        "embedding_pca"
    ]
    
    for node in nodes:
        if node in ["country", "embedding_pca"]:
            dot.node(node, node, style='filled', fillcolor='#FFB6C1')
        elif node in ['earnings']:
            dot.node(node, node, style='filled', fillcolor='#28b463')
        else:
            dot.node(node, node, style='filled', fillcolor='#FFFACD')
    
    # Edge definitions
    edges = [
        ("Full-Stack", "success_rate"), ("Full-Stack", "hour_rate_clean"), ("Full-Stack", "jobs"), ("Full-Stack", "hours"), ("Full-Stack", "certifs"),
        ("Front-End", "success_rate"), ("Front-End", "hour_rate_clean"), ("Front-End", "jobs"), ("Front-End", "hours"), ("Front-End", "certifs"),
        ("Back-End", "success_rate"), ("Back-End", "hour_rate_clean"), ("Back-End", "jobs"), ("Back-End", "hours"), ("Back-End", "certifs"),
        ("Other-Development", "success_rate"), ("Other-Development", "hour_rate_clean"), ("Other-Development", "jobs"), ("Other-Development", "hours"), ("Other-Development", "certifs"),
        ("Other-Tech", "success_rate"), ("Other-Tech", "hour_rate_clean"), ("Other-Tech", "jobs"), ("Other-Tech", "hours"), ("Other-Tech", "certifs"),
        ("Creative-Design", "success_rate"), ("Creative-Design", "hour_rate_clean"), ("Creative-Design", "jobs"), ("Creative-Design", "hours"), ("Creative-Design", "certifs"),
        ("Business-Marketing", "success_rate"), ("Business-Marketing", "hour_rate_clean"), ("Business-Marketing", "jobs"), ("Business-Marketing", "hours"), ("Business-Marketing", "certifs"),
        ("Other-Occupation", "success_rate"), ("Other-Occupation", "hour_rate_clean"), ("Other-Occupation", "jobs"), ("Other-Occupation", "hours"), ("Other-Occupation", "certifs"),
        ("portfolio", "success_rate"), ("portfolio", "hour_rate_clean"), ("portfolio", "jobs"), ("portfolio", "rate_cat"),
        ("certifs", "success_rate"), ("certifs", "jobs"), ("certifs", "hour_rate_clean"),
        ("video", "jobs"), ("video", "hours"),
        ("testims", "jobs"), ("testims", "hours"), ("testims", "rate_cat"),
        ("jobs", "hours"), ("jobs", "earnings"),
        ("hour_rate_clean", "earnings"),
        ("success_rate", "rate_cat"), ("success_rate", "hour_rate_clean"), ("success_rate", "hours"), ("success_rate", "jobs"),
        ("rate_cat", "jobs"), ("rate_cat", "hours"), ("rate_cat", "hour_rate_clean"),
        ("hours_per_week", "jobs"), ("hours_per_week", "hours"), ("hours_per_week", "earnings"),
        ("hours", "earnings"),
        ("num-occups", "jobs"), ("num-occups", "hours"), ("num-occups", "hour_rate_clean"), ("num-occups", "earnings"), 
        ("country", "hour_rate_clean"), ("country", "jobs"), ("country", "hours"), ("country", "portfolio"),
        ("country", "video"), ("country", "testims"), ("country", "certifs"), ("country", "hours_per_week"),
        ("country", "success_rate"),
        ("embedding_pca", "success_rate"), ("embedding_pca", "hour_rate_clean"), ("embedding_pca", "jobs"), ("embedding_pca", "hours"),
        ("embedding_pca", "earnings"), ("embedding_pca", "rate_cat"), ("embedding_pca", "certifs")
    ]
    
    for edge in edges:
        dot.edge(edge[0], edge[1])
    
    # Render the graph
    return dot

  st.graphviz_chart(create_freelance_dag())



with st.sidebar:    
    st.write('**Select Variables to Include**')
    include_country = st.checkbox('Country', value=True)
    include_video = st.checkbox('Video', value=True)
    include_portfolio = st.checkbox('Portfolio', value=True)
    include_success_rate = st.checkbox('Success Rate', value=True)
    include_embed = st.checkbox('Profile Quality', value=True)
    
    st.write('**Variable Values**')
    country = st.selectbox('Country', ('Egypt', 'India')) if include_country else None
    video = st.selectbox('Video', ('No', 'Yes')) if include_video else None
    portfolio = st.selectbox('Portfolio', ('No', 'Yes')) if include_portfolio else None
    success_rate = st.slider('Success Rate', min_value=0.5, max_value=1.0, value=0.9, step=0.1) if include_success_rate else None
    embedding_pca = st.slider('Profile Quality', min_value=-4, max_value=6, value=0, step=1) if include_embed else None

   
    # Encoding the inputs
    encoded_inputs = {}
    if include_country:
        encoded_inputs['country'] = 1 if country == 'Egypt' else 0
    if include_video:
        encoded_inputs['video'] = 1 if video == 'Yes' else 0
    if include_portfolio:
        encoded_inputs['portfolio'] = 1 if portfolio == 'Yes' else 0
    if include_success_rate:
        encoded_inputs['success_rate'] = success_rate
    if include_success_rate:
        encoded_inputs['embedding_pca'] = embedding_pca

# Loading
# Loading
@st.cache_resource  # This will cache the model to improve performance
def load_model():
    try:
        with gzip.open('model.pkl.gz', 'rb') as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model when the app starts
gcm_model = load_model()

if gcm_model is not None:
    st.success("Model loaded successfully!")
else:
    st.error("Failed to load the model. Please check if 'model-pbz2' exists in the correct location.")





