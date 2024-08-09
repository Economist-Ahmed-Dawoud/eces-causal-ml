import streamlit as st
import pandas as pd

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

  dag = create_freelance_dag()
  dag
