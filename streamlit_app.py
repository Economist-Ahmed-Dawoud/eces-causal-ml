import streamlit as st
import pandas as pd

st.title('🚀 Cuasal ML')

st.info('By leveraging counterfactual causal reasoning and advanced machine learning techniques, we seek to provide deep insights into the dynamics of the digital freelance economy.')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Economist-Ahmed-Dawoud/Data/main/streamlit-upwork-data.csv')
  df
  
