import streamlit as st
import pandas as pd
from graphviz import Digraph
import pickle
import requests
from io import BytesIO

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


@st.cache_resource
def load_model_from_drive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            content = response.content
            
            # Let's check the first few bytes of the content
            st.write(f"First 100 bytes of content: {content[:100]}")
            
            # If it's HTML, it will likely start with "<!DOCTYPE html>" or "<html>"
            if content.startswith(b'<!DOCTYPE html>') or content.startswith(b'<html>'):
                st.error("Received HTML instead of pickle data. The link might be incorrect or require authentication.")
                return None
            
            try:
                model = pickle.loads(content)
                return model
            except pickle.UnpicklingError:
                st.error("Failed to unpickle the data. The file might not be in pickle format.")
                return None
        else:
            st.error(f"Failed to load model. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Google Drive direct download link
drive_link = 'https://drive.usercontent.google.com/download?id=1BbYMeMs0kzW-Ng2RPAMupmnnip-gxmOw&export=download&authuser=0'

# Load the model
causal_model = load_model_from_drive(drive_link)

if causal_model is None:
    st.stop()  # Stop the app if model loading failed

# Rest of your Streamlit app code...
This code will show us the first 100 bytes of the content we're receiving, which should help us understand what we're actually getting from the link.

If we confirm that we're receiving HTML instead of pickle data, there could be a few reasons:
a. The link might require authentication.
b. The file might not be set to be publicly accessible.
c. Google Drive might be serving a download page instead of the file directly.
To address these issues, we might need to use a different approach to download from Google Drive. Here's an alternative method:

pythonCopyimport streamlit as st
import requests
import pickle
import re
from io import BytesIO

@st.cache_resource
def load_model_from_drive(file_id):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response):
        CHUNK_SIZE = 32768
        content = BytesIO()
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                content.write(chunk)
        content.seek(0)
        return content

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    content = save_response_content(response)

    try:
        model = pickle.load(content)
        return model
    except Exception as e:
        st.error(f"Error unpickling model: {str(e)}")
        return None

# Extract file ID from the link
file_id = "1BbYMeMs0kzW-Ng2RPAMupmnnip-gxmOw"

# Load the model
causal_model = load_model_from_drive(file_id)

if causal_model is None:
    st.stop()  # Stop the app if model loading failed
