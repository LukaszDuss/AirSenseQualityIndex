# -*- coding: utf-8 -*-
"""Material Icons — składnia Streamlit :material/nazwa: (1.33+) oraz HTML span."""


# --- Streamlit (:material/…: w przyciskach, caption, markdown) ---
AIR = ":material/air:"
ADMIN = ":material/admin_panel_settings:"
PERSON = ":material/person:"
REFRESH = ":material/refresh:"
DELETE = ":material/delete:"
ROCKET = ":material/rocket_launch:"
STAR = ":material/star:"
CHECK_CIRCLE = ":material/check_circle:"
ANALYTICS = ":material/analytics:"
EDIT_NOTE = ":material/edit_note:"
AUTO_AWESOME = ":material/auto_awesome:"
CHECK = ":material/check:"
LOCATION = ":material/location_on:"
THERMOSTAT = ":material/thermostat:"
SEARCH = ":material/search:"
ADD = ":material/add:"
WARNING = ":material/warning:"
CLOUD = ":material/cloud:"

# Wartości w tabelach audytu (zamiast ✅)
YES = CHECK
NO = "—"


def lbl(icon: str, text: str) -> str:
    """Ikona + tekst (przyciski, etykiety)."""
    return f"{icon} {text}".strip()


def badge(icon: str, text: str) -> str:
    """Krótka odznaka w tytule expandera."""
    return lbl(icon, text)


def html_icon(name: str, *, size_px: int = 20) -> str:
    """HTML dla nagłówków (page_header)."""
    return (
        f'<span class="material-icons" style="font-size:{size_px}px;'
        f'vertical-align:middle;margin-right:6px;">{name}</span>'
    )


def html_title(icon_name: str, title: str, *, size_px: int = 32) -> str:
    return f"{html_icon(icon_name, size_px=size_px)}{title}"
