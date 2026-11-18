import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("Test Folium")
m = folium.Map(location=[18.0, 73.7], zoom_start=8)
st_folium(m, use_container_width=True, height=400, key="test_map")
