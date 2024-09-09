from model_helper import generate_menu
import streamlit as st

st.title("Restaurant Name Generator")

with st.sidebar:
    cuisine = st.sidebar.selectbox("Pick a Cuisine",("Indian","Italian","Mexican","French"));


if cuisine:
    result= generate_menu(cuisine);
    st.header(result['restaurant_name'])
    menu_items= result['menu_items']
    st.write("**Menu Items**")
    for item in menu_items:
        st.write(item)