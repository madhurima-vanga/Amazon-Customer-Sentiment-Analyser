import streamlit as st
from welcome_page import main as welcome_page
from main_app import main_app

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Welcome"

# Navigation logic
if st.session_state.page == "Welcome":
    welcome_page()
elif st.session_state.page == "Main":
    main_app()
