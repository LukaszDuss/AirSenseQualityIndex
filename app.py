# -*- coding: utf-8 -*-
"""AirSense — / (klient), /admin (panel)"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st

import env_config
import app_pages
import ui_common as ui

env_config.load_env()
st.set_page_config(
    page_title="AirSense Quality AI",
    page_icon=":material/air:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject_css()

client_pg, admin_pg = app_pages.build_pages()
ui.set_nav_pages(client_pg, admin_pg)

pg = st.navigation([client_pg, admin_pg], position="hidden")
pg.run()
