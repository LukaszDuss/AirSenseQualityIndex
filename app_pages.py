# -*- coding: utf-8 -*-
"""Definicja stron multipage (/, /admin) — współdzielone przez app.py i nawigację."""
import streamlit as st

import stations_settings
import ui_common as ui
import ui_icons as mi
from views import admin_router, client_page

_client_page = None
_admin_page = None


def _render_client():
    stations = stations_settings.enabled_stations()
    if not stations:
        ui.page_header("Brak skonfigurowanych stacji")
        st.warning("Włącz stacje w panelu administracyjnym (/admin → Ustawienia).")
        ui.nav_to_admin(mi.lbl(mi.ADMIN, "Przejdź do panelu admin"))
        return
    client_page.render(stations)


def _render_admin():
    admin_router.render()


def build_pages():
    global _client_page, _admin_page
    _client_page = st.Page(
        _render_client,
        title="AirSense",
        icon=":material/air:",
        default=True,
    )
    _admin_page = st.Page(
        _render_admin,
        title="Admin",
        icon=":material/admin_panel_settings:",
        url_path="admin",
    )
    return _client_page, _admin_page


def get_admin_page():
    return _admin_page


def get_client_page():
    return _client_page
