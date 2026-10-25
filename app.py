import hashlib
import hmac
import html
import json
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Traffic Flow Prediction System",
    page_icon="🚦",
    layout="wide"
)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]
CITIES = {"Pune": 1.0, "Mumbai": 1.6, "Nagpur": 0.7,
          "Nashik": 0.6, "Delhi": 1.8}
WEATHER = ["Clear", "Cloudy", "Rainy"]
LEVELS = ["Low", "Medium", "High"]
LEVEL_COLORS = {"Low": "#22c55e", "Medium": "#facc15", "High": "#ef4444"}
LEVEL_RGB = {"Low": [34, 197, 94], "Medium": [250, 204, 21],
             "High": [239, 68, 68]}
LEVEL_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
DAY_FACTOR = {"Monday": 1.05, "Tuesday": 1.0, "Wednesday": 1.02,
              "Thursday": 1.04, "Friday": 1.12, "Saturday": 0.9,
              "Sunday": 0.75}
WEATHER_FACTOR = {"Clear": 1.0, "Cloudy": 0.97, "Rainy": 0.9}
CITY_CODE = {c: i for i, c in enumerate(CITIES)}
DAY_CODE = {d: i for i, d in enumerate(DAYS)}
WEATHER_CODE = {w: i for i, w in enumerate(WEATHER)}
FEATURES = ["city_code", "day_code", "hour", "weather_code"]
FEATURE_LABELS = ["City", "Day", "Hour", "Weather"]
DAY_COLORS = ["#38bdf8", "#818cf8", "#a78bfa", "#f472b6",
              "#fb923c", "#facc15", "#4ade80"]
CITY_COLORS = ["#38bdf8", "#f472b6", "#facc15", "#4ade80", "#fb923c"]
UPLOAD_COLS = ["city", "day", "hour", "weather", "vehicles", "speed"]
IST = timezone(timedelta(hours=5, minutes=30))

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# City centre: latitude, longitude, map zoom
CITY_VIEW = {
    "Pune": (18.5204, 73.8567, 10.5),
    "Mumbai": (19.0760, 72.8777, 10.0),
    "Nagpur": (21.1458, 79.0882, 11.0),
    "Nashik": (19.9975, 73.7898, 11.0),
    "Delhi": (28.6139, 77.2090, 10.0),
}

# Sample roads / areas: name, latitude, longitude, busy factor
# (coordinates are approximate, the busy factor is an estimate)
ROADS = {
    "Pune": [
        ("Shivajinagar", 18.5308, 73.8475, 1.15),
        ("Swargate", 18.5018, 73.8636, 1.20),
        ("Hinjewadi", 18.5912, 73.7389, 1.25),
        ("Kothrud", 18.5074, 73.8077, 1.00),
        ("Hadapsar", 18.5089, 73.9260, 1.05),
        ("Viman Nagar", 18.5679, 73.9143, 0.95),
        ("Baner", 18.5590, 73.7868, 1.05),
        ("Katraj", 18.4575, 73.8677, 0.85),
        ("Pimpri-Chinchwad", 18.6279, 73.8009, 0.90),
        ("Kharadi", 18.5515, 73.9348, 1.10),
    ],
    "Mumbai": [
        ("Bandra", 19.0596, 72.8295, 1.20),
        ("Andheri", 19.1136, 72.8697, 1.25),
        ("Dadar", 19.0178, 72.8478, 1.20),
        ("Colaba", 18.9067, 72.8147, 0.85),
        ("Borivali", 19.2307, 72.8567, 1.00),
        ("Powai", 19.1176, 72.9060, 1.05),
        ("Kurla", 19.0726, 72.8845, 1.20),
        ("Worli", 19.0176, 72.8174, 0.95),
        ("Thane", 19.2183, 72.9781, 1.00),
        ("Vashi", 19.0771, 72.9986, 0.80),
    ],
    "Nagpur": [
        ("Sitabuldi", 21.1463, 79.0849, 1.20),
        ("Sadar", 21.1613, 79.0793, 1.05),
        ("Dharampeth", 21.1330, 79.0630, 1.10),
        ("Itwari", 21.1450, 79.1120, 1.10),
        ("Manish Nagar", 21.0920, 79.0630, 0.85),
        ("Khamla", 21.1190, 79.0700, 0.95),
        ("Civil Lines", 21.1568, 79.0780, 0.90),
        ("Wardha Road", 21.1040, 79.0560, 1.00),
    ],
    "Nashik": [
        ("Nashik Road", 19.9513, 73.8365, 1.15),
        ("College Road", 20.0068, 73.7589, 1.10),
        ("Panchavati", 20.0064, 73.7910, 1.05),
        ("CBS Junction", 19.9993, 73.7797, 1.20),
        ("Gangapur Road", 20.0100, 73.7450, 1.00),
        ("Satpur", 20.0000, 73.7300, 0.95),
        ("Dwarka Circle", 19.9990, 73.8085, 1.15),
        ("Indira Nagar", 19.9840, 73.7930, 0.90),
    ],
    "Delhi": [
        ("Connaught Place", 28.6315, 77.2167, 1.15),
        ("ITO", 28.6289, 77.2410, 1.25),
        ("Karol Bagh", 28.6519, 77.1909, 1.15),
        ("Dwarka", 28.5921, 77.0460, 0.90),
        ("Rohini", 28.7495, 77.0565, 0.95),
        ("Saket", 28.5245, 77.2066, 1.00),
        ("Nehru Place", 28.5494, 77.2501, 1.10),
        ("Lajpat Nagar", 28.5677, 77.2433, 1.10),
        ("Janakpuri", 28.6219, 77.0878, 0.90),
        ("Vasant Kunj", 28.5200, 77.1590, 0.80),
    ],
}

# Themes (the Change Theme button cycles through these)
THEMES = {
    "🌙 Dark": dict(
        bg1="#0f172a", bg2="#1e293b", text="#e2e8f0", accent="#38bdf8",
        card="rgba(255,255,255,0.07)", border="rgba(255,255,255,0.14)",
        side1="#4c1d95", side2="#1e3a8a", side_text="#ffffff",
        sel_bg="rgba(255,255,255,0.15)", btn1="#6366f1", btn2="#06b6d4",
        chart_bg="#1e293b", chart_grid="#475569", dark_map="1"),
    "☀️ Light": dict(
        bg1="#f8fafc", bg2="#e2e8f0", text="#0f172a", accent="#2563eb",
        card="rgba(255,255,255,0.85)", border="rgba(15,23,42,0.15)",
        side1="#dbeafe", side2="#e0e7ff", side_text="#0f172a",
        sel_bg="rgba(255,255,255,0.7)", btn1="#2563eb", btn2="#0ea5e9",
        chart_bg="#ffffff", chart_grid="#cbd5e1", dark_map="0"),
    "🌊 Ocean": dict(
        bg1="#042f2e", bg2="#134e4a", text="#ccfbf1", accent="#2dd4bf",
        card="rgba(255,255,255,0.08)", border="rgba(255,255,255,0.16)",
        side1="#0f766e", side2="#155e75", side_text="#ffffff",
        sel_bg="rgba(255,255,255,0.15)", btn1="#0d9488", btn2="#0891b2",
        chart_bg="#134e4a", chart_grid="#2f6f68", dark_map="1"),
    "🌇 Sunset": dict(
        bg1="#2a0a18", bg2="#4a1d2f", text="#ffe4e6", accent="#fb923c",
        card="rgba(255,255,255,0.08)", border="rgba(255,255,255,0.16)",
        side1="#9d174d", side2="#c2410c", side_text="#ffffff",
        sel_bg="rgba(255,255,255,0.18)", btn1="#e11d48", btn2="#f97316",
        chart_bg="#4a1d2f", chart_grid="#7f3a50", dark_map="1"),
}
THEME_NAMES = list(THEMES.keys())

# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "name" not in st.session_state:
    st.session_state.name = ""
if "user_key" not in st.session_state:
    st.session_state.user_key = None
if "theme" not in st.session_state or st.session_state.theme not in THEMES:
    st.session_state.theme = THEME_NAMES[0]

T = THEMES[st.session_state.theme]


# ---------------------------------------------------------------
# Accounts (stored in users.json with salted, hashed passwords)
# ---------------------------------------------------------------
def load_users():
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_users(users):
    try:
        tmp_path = USERS_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        os.replace(tmp_path, USERS_FILE)
        return True
    except OSError:
        return False


def hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 150000
    ).hex()


def create_account(username, password, confirm):
    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]{3,20}", username):
        return False, "Username must be 3 to 20 characters (letters, numbers or underscore)."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm:
        return False, "Passwords do not match."
    users = load_users()
    key = username.lower()
    if key in users:
        return False, "This username is already taken. Please choose another."
    salt = secrets.token_hex(16)
    users[key] = {
        "display": username,
        "salt": salt,
        "hash": hash_password(password, salt),
        "theme": st.session_state.theme,
    }
    if not save_users(users):
        return False, "Could not save the account on this computer."
    return True, key


def check_login(username, password):
    key = username.strip().lower()
    record = load_users().get(key)
    if not record:
        return None
    try:
        expected = hash_password(password, record["salt"])
    except (KeyError, ValueError, TypeError):
        return None
    if hmac.compare_digest(expected, str(record.get("hash", ""))):
        return key
    return None


def start_session(user_key, display_name):
    st.session_state.logged_in = True
    st.session_state.name = display_name
    st.session_state.user_key = user_key
    if user_key:
        saved_theme = load_users().get(user_key, {}).get("theme")
        if saved_theme in THEMES:
            st.session_state.theme = saved_theme


def logout():
    st.session_state.logged_in = False
    st.session_state.name = ""
    st.session_state.user_key = None
    for k in ("last_pred", "custom_models", "use_custom", "custom_rows"):
        st.session_state.pop(k, None)


def cycle_theme():
    i = THEME_NAMES.index(st.session_state.theme)
    st.session_state.theme = THEME_NAMES[(i + 1) % len(THEME_NAMES)]
    key = st.session_state.get("user_key")
    if key:
        users = load_users()
        if key in users:
            users[key]["theme"] = st.session_state.theme
            save_users(users)


# ---------------------------------------------------------------
# Styling
# ---------------------------------------------------------------
CSS_TEMPLATE = """
.stApp { background: linear-gradient(135deg, __BG1__ 0%, __BG2__ 100%); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-left: 3rem; padding-right: 3rem; max-width: 100%; }
@media (max-width: 640px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
.block-container h1, .block-container h2, .block-container h3,
.block-container h4, .block-container label,
.block-container p, .block-container li { color: __TEXT__ !important; }
.block-container [data-testid="stSlider"] div { color: __TEXT__ !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, __SIDE1__ 0%, __SIDE2__ 100%); }
[data-testid="stSidebar"] * { color: __SIDE_TEXT__ !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div { background: __SEL_BG__; border-radius: 10px; }
.card { background: __CARD__; border: 1px solid __BORDER__; border-radius: 18px; padding: 22px; text-align: center; margin-bottom: 14px; }
.card h4 { margin: 0; font-size: 1rem; }
.card h2 { margin: 8px 0; font-size: 2rem; color: __ACCENT__ !important; }
.card p { margin: 0; font-size: 0.9rem; opacity: 0.85; }
.box { border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; color: __TEXT__ !important; font-size: 1rem; }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button { width: 100%; border-radius: 12px; background: linear-gradient(90deg, __BTN1__, __BTN2__); color: white; border: none; font-weight: 600; padding: 0.6rem 1rem; }
.block-container .stButton > button p, .block-container .stFormSubmitButton > button p, .block-container .stDownloadButton > button p { color: #ffffff !important; }
[data-testid="stSidebar"] .stButton > button * { color: #ffffff !important; }
.linkrow { display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0 14px 0; }
.linkbtn { display: inline-block; padding: 0.55rem 1.1rem; border-radius: 12px; color: #ffffff !important; text-decoration: none !important; font-weight: 600; font-size: 0.95rem; }
"""


def apply_css():
    css = CSS_TEMPLATE
    for key, value in T.items():
        css = css.replace("__" + key.upper() + "__", value)
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


apply_css()

# Hide sidebar on login page
if not st.session_state.logged_in:
    st.markdown("""
    <style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def now_ist():
    return datetime.now(IST)


def pattern(h, weekend):
    if not weekend:
        return (60 + 380 * np.exp(-((h - 9) ** 2) / 6)
                + 450 * np.exp(-((h - 18) ** 2) / 7)
                + 150 * np.exp(-((h - 13) ** 2) / 10))
    return (50 + 250 * np.exp(-((h - 13) ** 2) / 20)
            + 280 * np.exp(-((h - 19) ** 2) / 10))


def level_of(value, scale):
    v = value / scale
    if v < 200:
        return "Low"
    elif v < 400:
        return "Medium"
    return "High"


@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    rows = []
    for city_name, scale_value in CITIES.items():
        for week in range(1, 9):
            for day_name in DAYS:
                weekend = day_name in ("Saturday", "Sunday")
                for hour in range(24):
                    base = pattern(hour, weekend)
                    weather = str(rng.choice(WEATHER, p=[0.55, 0.30, 0.15]))
                    vehicles = (base * scale_value * DAY_FACTOR[day_name]
                                * WEATHER_FACTOR[weather]
                                * rng.normal(1, 0.07))
                    vehicles = int(max(5, vehicles))
                    speed = 58 - vehicles / (scale_value * 12) + rng.normal(0, 2)
                    if weather == "Rainy":
                        speed -= 4
                    speed = round(float(np.clip(speed, 8, 60)), 1)
                    rows.append([city_name, week, day_name, hour, weather,
                                 vehicles, speed])
    data = pd.DataFrame(
        rows,
        columns=["city", "week", "day", "hour", "weather",
                 "vehicles", "speed"]
    )
    data["level"] = [level_of(v, CITIES[c])
                     for v, c in zip(data["vehicles"], data["city"])]
    data["city_code"] = data["city"].map(CITY_CODE)
    data["day_code"] = data["day"].map(DAY_CODE)
    data["weather_code"] = data["weather"].map(WEATHER_CODE)
    return data


@st.cache_resource
def train_models():
    data = load_data()
    X = data[FEATURES]
    vehicle_model = RandomForestRegressor(n_estimators=100, random_state=42)
    vehicle_model.fit(X, data["vehicles"])
    speed_model = RandomForestRegressor(n_estimators=100, random_state=42)
    speed_model.fit(X, data["speed"])
    return vehicle_model, speed_model


@st.cache_data
def evaluate_models():
    data = load_data()
    X = data[FEATURES]
    X_tr, X_te, v_tr, v_te, s_tr, s_te = train_test_split(
        X, data["vehicles"], data["speed"], test_size=0.2, random_state=42
    )
    vm = RandomForestRegressor(n_estimators=100, random_state=42)
    vm.fit(X_tr, v_tr)
    sm = RandomForestRegressor(n_estimators=100, random_state=42)
    sm.fit(X_tr, s_tr)
    v_pred = vm.predict(X_te)
    s_pred = sm.predict(X_te)
    cities = data.loc[X_te.index, "city"].to_numpy()
    true_levels = data.loc[X_te.index, "level"].to_numpy()
    pred_levels = np.array([level_of(v, CITIES[c])
                            for v, c in zip(v_pred, cities)])
    return {
        "v_r2": float(r2_score(v_te, v_pred)),
        "v_mae": float(mean_absolute_error(v_te, v_pred)),
        "s_r2": float(r2_score(s_te, s_pred)),
        "s_mae": float(mean_absolute_error(s_te, s_pred)),
        "level_acc": float((true_levels == pred_levels).mean() * 100),
        "v_true": v_te.to_numpy(),
        "v_pred": v_pred,
        "importance": vm.feature_importances_,
        "test_rows": int(len(X_te)),
        "train_rows": int(len(X_tr)),
    }


vehicle_model = None
speed_model = None


def get_models():
    custom = st.session_state.get("custom_models")
    if st.session_state.get("use_custom") and custom:
        return custom
    return vehicle_model, speed_model


def predict_frame(city_name, day_name, weather_name, hours):
    hours = list(hours)
    return pd.DataFrame({
        "city_code": [CITY_CODE[city_name]] * len(hours),
        "day_code": [DAY_CODE[day_name]] * len(hours),
        "hour": hours,
        "weather_code": [WEATHER_CODE[weather_name]] * len(hours)
    })[FEATURES]


def predict_day(city_name, day_name, weather_name, hours=None):
    """Return predicted vehicles and speed arrays for the given hours."""
    if hours is None:
        hours = range(24)
    vm, sm = get_models()
    X = predict_frame(city_name, day_name, weather_name, hours)
    vehicles = np.maximum(vm.predict(X), 0)
    speeds = np.clip(sm.predict(X), 8, 60)
    return vehicles, speeds


def rush_hours(city_name, day_name, weather_name="Clear"):
    vehicles, _ = predict_day(city_name, day_name, weather_name)
    sc = CITIES[city_name]
    high = [h for h in range(24) if level_of(vehicles[h], sc) == "High"]
    ranges = []
    start = prev = None
    for h in high:
        if start is None:
            start = prev = h
        elif h == prev + 1:
            prev = h
        else:
            ranges.append((start, prev))
            start = prev = h
    if start is not None:
        ranges.append((start, prev))
    return [f"{a:02d}:00 to {(b + 1) % 24:02d}:00" for a, b in ranges]


def style_chart(fig, ax):
    fig.patch.set_facecolor(T["chart_bg"])
    ax.set_facecolor(T["chart_bg"])
    ax.tick_params(colors=T["text"])
    ax.xaxis.label.set_color(T["text"])
    ax.yaxis.label.set_color(T["text"])
    ax.title.set_color(T["text"])
    for spine in ax.spines.values():
        spine.set_color(T["chart_grid"])
    ax.grid(alpha=0.2, color=T["chart_grid"])


def show(fig):
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def fmt_hour(h):
    return f"{int(h):02d}:00"


def card(title, value, sub="", col=None):
    target = col if col is not None else st
    target.markdown(
        f"<div class='card'><h4>{title}</h4><h2>{value}</h2><p>{sub}</p></div>",
        unsafe_allow_html=True
    )


def box(text, color="#6366f1"):
    st.markdown(
        f"<div class='box' style='background:{color}33; "
        f"border-left:5px solid {color};'>{text}</div>",
        unsafe_allow_html=True
    )


def link_btn(label, url, color="#6366f1"):
    safe_url = html.escape(url, quote=True)
    return (f"<a class='linkbtn' href='{safe_url}' target='_blank' "
            f"rel='noopener noreferrer' style='background:{color};'>{label}</a>")


def share_buttons_html(message, url=""):
    full = (message + " " + url).strip()
    parts = [link_btn("🟢 WhatsApp", "https://wa.me/?text=" + quote(full), "#16a34a")]
    if url:
        parts.append(link_btn(
            "✈️ Telegram",
            "https://t.me/share/url?url=" + quote(url, safe="") + "&text=" + quote(message),
            "#0284c7"))
    parts.append(link_btn(
        "✉️ Email",
        "mailto:?subject=" + quote("Traffic Flow Prediction") + "&body=" + quote(full),
        "#f59e0b"))
    if url:
        parts.append(link_btn(
            "𝕏 Post",
            "https://twitter.com/intent/tweet?text=" + quote(message) + "&url=" + quote(url, safe=""),
            "#334155"))
        parts.append(link_btn(
            "💼 LinkedIn",
            "https://www.linkedin.com/sharing/share-offsite/?url=" + quote(url, safe=""),
            "#0a66c2"))
    return "<div class='linkrow'>" + "".join(parts) + "</div>"


def show_links(html_text):
    st.markdown(html_text, unsafe_allow_html=True)


def live_traffic_url(city_name):
    lat, lon, _ = CITY_VIEW[city_name]
    return f"https://www.google.com/maps/@{lat},{lon},12z/data=!5m1!1e1"


def show_map(map_df, lat, lon, zoom):
    try:
        import pydeck as pdk
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df[["name", "lat", "lon", "level", "vehicles", "speed", "color"]],
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=700,
            radius_min_pixels=9,
            radius_max_pixels=28,
            pickable=True,
            opacity=0.9,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
        )
        view = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom)
        tooltip = {
            "html": "<b>{name}</b><br/>{level} traffic<br/>"
                    "{vehicles} vehicles/hour<br/>{speed} km/h",
            "style": {"color": "white"},
        }
        style = "dark" if T["dark_map"] == "1" else "light"
        try:
            deck = pdk.Deck(layers=[layer], initial_view_state=view,
                            tooltip=tooltip, map_provider="carto",
                            map_style=style)
        except TypeError:
            deck = pdk.Deck(layers=[layer], initial_view_state=view,
                            tooltip=tooltip)
        st.pydeck_chart(deck)
    except Exception:
        st.map(map_df[["lat", "lon"]])


def clean_uploaded(raw):
    raw = raw.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in UPLOAD_COLS if c not in raw.columns]
    if missing:
        return None, "Missing column(s): " + ", ".join(missing)
    data = raw[UPLOAD_COLS].copy()
    data["city"] = data["city"].astype(str).str.strip().str.title()
    abbr = {d[:3].lower(): d for d in DAYS}
    data["day"] = data["day"].astype(str).str.strip().str.lower().str[:3].map(abbr)
    data["weather"] = data["weather"].astype(str).str.strip().str.title()
    for col in ("hour", "vehicles", "speed"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    data = data[
        data["city"].isin(list(CITIES))
        & data["weather"].isin(WEATHER)
        & data["hour"].between(0, 23)
        & (data["vehicles"] >= 0)
        & data["speed"].between(1, 150)
    ].copy()
    if len(data) < 50:
        return None, f"Only {len(data)} valid rows found. At least 50 rows are needed."
    data["hour"] = data["hour"].astype(int)
    return data.reset_index(drop=True), ""


def train_custom(data):
    d2 = data.copy()
    d2["city_code"] = d2["city"].map(CITY_CODE)
    d2["day_code"] = d2["day"].map(DAY_CODE)
    d2["weather_code"] = d2["weather"].map(WEATHER_CODE)
    X = d2[FEATURES]
    vm = RandomForestRegressor(n_estimators=100, random_state=42)
    vm.fit(X, d2["vehicles"])
    sm = RandomForestRegressor(n_estimators=100, random_state=42)
    sm.fit(X, d2["speed"])
    return vm, sm


# ---------------------------------------------------------------
# Login / Create account page
# ---------------------------------------------------------------
if not st.session_state.logged_in:
    st.write("")
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown(
            "<div style='text-align:center; font-size:80px;'>🚦</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<h1 style='text-align:center;'>Traffic Flow Prediction</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center;'>Smart city traffic analysis and prediction</p>",
            unsafe_allow_html=True
        )
        st.button("🎨 Change Theme  (" + st.session_state.theme + ")",
                  on_click=cycle_theme, key="theme_btn_login")
        st.write("")

        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Create Account"])

        with tab_login:
            with st.form("login_form"):
                l_user = st.text_input("Username", placeholder="Your username")
                l_pass = st.text_input("Password", type="password",
                                       placeholder="Your password")
                l_submit = st.form_submit_button("Login 🚀")
            if l_submit:
                if not l_user.strip() or not l_pass:
                    box("Please enter your username and password.", "#ef4444")
                else:
                    user_key = check_login(l_user, l_pass)
                    if user_key:
                        display = load_users()[user_key].get("display", l_user.strip())
                        start_session(user_key, display)
                        st.rerun()
                    else:
                        box("Wrong username or password.", "#ef4444")

        with tab_signup:
            with st.form("signup_form"):
                s_user = st.text_input("Choose a username",
                                       placeholder="3 to 20 letters or numbers")
                s_pass = st.text_input("Choose a password", type="password",
                                       placeholder="At least 6 characters")
                s_pass2 = st.text_input("Confirm password", type="password",
                                        placeholder="Type the password again")
                s_submit = st.form_submit_button("Create Account ✅")
            if s_submit:
                ok, result = create_account(s_user, s_pass, s_pass2)
                if ok:
                    start_session(result, s_user.strip())
                    st.rerun()
                else:
                    box(result, "#ef4444")

        st.write("")
        if st.button("👤 Continue as Guest", key="guest_btn"):
            start_session(None, "Guest")
            st.rerun()
    st.stop()

# ---------------------------------------------------------------
# Load data and models (after login)
# ---------------------------------------------------------------
df = load_data()
vehicle_model, speed_model = train_models()

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("🚦 Traffic System")
st.sidebar.write(f"👤 Welcome, **{st.session_state.name}**")
if st.session_state.user_key is None:
    st.sidebar.caption("Guest mode: create an account to save your theme.")
city = st.sidebar.selectbox("🏙️ Select City", list(CITIES.keys()))
page = st.sidebar.radio(
    "📂 Menu",
    ["🏠 Home", "📊 Dashboard", "📅 Weekly Analysis",
     "🏙️ City Comparison", "🔮 Traffic Prediction",
     "⚖️ Compare Scenarios", "🧭 Trip Planner", "🗺️ Traffic Map",
     "🎯 Model Accuracy", "📤 Upload Data", "🗂️ Dataset",
     "🔗 Share App", "ℹ️ About"]
)
st.sidebar.button("🎨 Change Theme", on_click=cycle_theme,
                  key="theme_btn_sidebar")
st.sidebar.caption("Current theme: " + st.session_state.theme)
if st.session_state.get("use_custom"):
    st.sidebar.caption("🧪 Using the model trained on your uploaded data")
if st.sidebar.button("Logout"):
    logout()
    st.rerun()

scale = CITIES[city]
d = df[df["city"] == city]

# Main heading
st.title("🚦 Advanced Traffic Flow Prediction System")
st.write(f"Machine Learning based traffic analysis for **{city}**.")
st.divider()

# ---------------------------------------------------------------
# Home page
# ---------------------------------------------------------------
if page == "🏠 Home":
    st.subheader(f"👋 Hello, {st.session_state.name}!")
    st.write(
        "Explore hourly traffic, weekly patterns, city comparisons, "
        "smart predictions and the best time to travel in five major cities."
    )

    now = now_ist()
    today = DAYS[now.weekday()]
    v_now, s_now = predict_day(city, today, "Clear", [now.hour])
    lvl_now = level_of(v_now[0], scale)

    st.subheader(f"⏱️ Live Status: {city}")
    box(f"{LEVEL_EMOJI[lvl_now]} <b>{lvl_now} traffic</b> right now "
        f"({today}, {now.strftime('%H:%M')} IST): about {v_now[0]:.0f} "
        f"vehicles per hour at {s_now[0]:.0f} km/h (assuming clear weather)",
        LEVEL_COLORS[lvl_now])
    alerts = rush_hours(city, today)
    if alerts:
        box(f"🚨 <b>Rush hour alert</b> for today ({today}): heavy traffic "
            f"expected during {', '.join(alerts)}", "#ef4444")
    else:
        box(f"✅ No heavy traffic expected in {city} today ({today}).", "#22c55e")

    hourly = d.groupby("hour")["vehicles"].mean()
    high_share = (d["level"] == "High").mean() * 100

    st.subheader(f"📍 {city} at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    card("Average Vehicles", f"{d['vehicles'].mean():.0f}", "per hour", c1)
    card("Average Speed", f"{d['speed'].mean():.0f} km/h", "all roads", c2)
    card("Busiest Hour", fmt_hour(hourly.idxmax()), "daily rush", c3)
    card("High Traffic", f"{high_share:.1f}%", "of all hours", c4)

    st.subheader("🏙️ All Cities")
    city_avg = df.groupby("city")["vehicles"].mean().reindex(list(CITIES))
    cols = st.columns(5)
    for i, (cname, avg) in enumerate(city_avg.items()):
        card(cname, f"{avg:.0f}", "avg vehicles per hour", cols[i])

    st.subheader("🧭 What You Can Do Here")
    box("📊 <b>Dashboard</b>: hourly traffic, speed, weather impact and key insights", "#6366f1")
    box("📅 <b>Weekly Analysis</b>: Monday to Sunday patterns with a heatmap", "#06b6d4")
    box("🏙️ <b>City Comparison</b>: compare Pune, Mumbai, Nagpur, Nashik and Delhi", "#a855f7")
    box("🔮 <b>Traffic Prediction</b>: predict vehicles and speed for any hour, day and weather", "#22c55e")
    box("⚖️ <b>Compare Scenarios</b>: compare two situations side by side, like clear vs rainy", "#14b8a6")
    box("🧭 <b>Trip Planner</b>: find the best time to leave and open the route in Google Maps", "#f97316")
    box("🗺️ <b>Traffic Map</b>: see which areas are free and which are jammed", "#ec4899")
    box("🎯 <b>Model Accuracy</b>: R², error and feature importance of the models", "#0ea5e9")
    box("📤 <b>Upload Data</b>: train the model on your own real traffic data", "#84cc16")
    box("🗂️ <b>Dataset</b>: filter, view and download the full dataset", "#f59e0b")
    box("🔗 <b>Share App</b>: send this app to friends and family", "#8b5cf6")

# ---------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------
elif page == "📊 Dashboard":
    st.subheader(f"📊 Traffic Overview: {city}")

    hourly = d.groupby("hour")["vehicles"].mean()
    hourly_speed = d.groupby("hour")["speed"].mean()
    peak_hour = int(hourly.idxmax())
    quiet_hour = int(hourly.idxmin())

    c1, c2, c3, c4 = st.columns(4)
    card("Total Records", f"{len(d):,}", "hourly readings", c1)
    card("Average Vehicles", f"{d['vehicles'].mean():.0f}", "per hour", c2)
    card("Peak Traffic", f"{d['vehicles'].max()}", "highest reading", c3)
    card("Busiest Hour", fmt_hour(peak_hour),
         f"{hourly.max():.0f} vehicles on average", c4)

    left, right = st.columns(2)

    with left:
        st.subheader("Traffic by Hour")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(hourly.index, hourly.values, color="#38bdf8")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Vehicles")
        ax.set_title("Hourly Traffic Analysis")
        style_chart(fig, ax)
        show(fig)

    with right:
        st.subheader("Traffic Level Distribution")
        counts = d["level"].value_counts().reindex(LEVELS).fillna(0)
        counts = counts[counts > 0]
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=[LEVEL_COLORS[i] for i in counts.index],
            textprops={"color": T["text"]}
        )
        fig2.patch.set_facecolor(T["chart_bg"])
        show(fig2)

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Average Speed by Hour")
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        ax3.plot(hourly_speed.index, hourly_speed.values,
                 marker="o", linewidth=2.5, color="#f472b6")
        ax3.set_xlabel("Hour of Day")
        ax3.set_ylabel("Speed (km/h)")
        ax3.set_title("Speed Through the Day")
        style_chart(fig3, ax3)
        show(fig3)

    with right2:
        st.subheader("Traffic by Weather")
        weather_avg = d.groupby("weather")["vehicles"].mean().reindex(WEATHER)
        fig4, ax4 = plt.subplots(figsize=(8, 4.5))
        ax4.bar(weather_avg.index, weather_avg.values,
                color=["#facc15", "#94a3b8", "#38bdf8"])
        ax4.set_ylabel("Average Vehicles")
        ax4.set_title("Vehicles in Different Weather")
        style_chart(fig4, ax4)
        show(fig4)

    st.subheader("💡 Key Insights")
    day_avg = d.groupby("day")["vehicles"].mean().reindex(DAYS)
    speed_weather = d.groupby("weather")["speed"].mean()
    high_share = (d["level"] == "High").mean() * 100

    box(f"🔴 Busiest hour: {fmt_hour(peak_hour)} with about {hourly.max():.0f} vehicles", "#ef4444")
    box(f"🟢 Quietest hour: {fmt_hour(quiet_hour)} with about {hourly.min():.0f} vehicles", "#22c55e")
    box(f"📅 Busiest day: {day_avg.idxmax()} (about {day_avg.max():.0f} vehicles per hour)", "#6366f1")
    box(f"📆 Quietest day: {day_avg.idxmin()} (about {day_avg.min():.0f} vehicles per hour)", "#06b6d4")
    box(f"🌧️ Average speed: Clear {speed_weather['Clear']:.0f} km/h, Rainy {speed_weather['Rainy']:.0f} km/h", "#f59e0b")
    box(f"🚨 High traffic occurs in {high_share:.1f}% of all hours", "#ec4899")

# ---------------------------------------------------------------
# Weekly analysis page
# ---------------------------------------------------------------
elif page == "📅 Weekly Analysis":
    st.subheader(f"📅 Weekly Analysis: {city}")

    day_avg = d.groupby("day")["vehicles"].mean().reindex(DAYS)
    day_speed = d.groupby("day")["speed"].mean().reindex(DAYS)
    weekend_days = ["Saturday", "Sunday"]
    weekday_avg = d[~d["day"].isin(weekend_days)]["vehicles"].mean()
    weekend_avg = d[d["day"].isin(weekend_days)]["vehicles"].mean()

    c1, c2, c3, c4 = st.columns(4)
    card("Busiest Day", day_avg.idxmax(), f"{day_avg.max():.0f} vehicles per hour", c1)
    card("Quietest Day", day_avg.idxmin(), f"{day_avg.min():.0f} vehicles per hour", c2)
    card("Mon to Fri Average", f"{weekday_avg:.0f}", "vehicles per hour", c3)
    card("Sat and Sun Average", f"{weekend_avg:.0f}", "vehicles per hour", c4)

    left, right = st.columns(2)

    with left:
        st.subheader("Average Vehicles by Day")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([x[:3] for x in DAYS], day_avg.values, color=DAY_COLORS)
        ax.set_ylabel("Average Vehicles")
        ax.set_title("Day-wise Traffic")
        style_chart(fig, ax)
        show(fig)

    with right:
        st.subheader("Average Speed by Day")
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.bar([x[:3] for x in DAYS], day_speed.values, color=DAY_COLORS)
        ax2.set_ylabel("Speed (km/h)")
        ax2.set_title("Day-wise Speed")
        style_chart(fig2, ax2)
        show(fig2)

    st.subheader("🔥 Traffic Heatmap (Day vs Hour)")
    pivot = d.pivot_table(index="day", columns="hour",
                          values="vehicles", aggfunc="mean").reindex(DAYS)
    fig3, ax3 = plt.subplots(figsize=(12, 4.5))
    im = ax3.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax3.set_xticks(range(24))
    ax3.set_yticks(range(7))
    ax3.set_yticklabels(DAYS)
    ax3.set_xlabel("Hour of Day")
    ax3.set_title("Average Vehicles")
    style_chart(fig3, ax3)
    ax3.grid(False)
    cbar = fig3.colorbar(im, ax=ax3)
    cbar.ax.tick_params(colors=T["text"])
    show(fig3)

    st.subheader("📈 Hourly Pattern for All Days")
    fig4, ax4 = plt.subplots(figsize=(12, 4.5))
    for i, day_name in enumerate(DAYS):
        curve = d[d["day"] == day_name].groupby("hour")["vehicles"].mean()
        ax4.plot(curve.index, curve.values, linewidth=2,
                 color=DAY_COLORS[i], label=day_name[:3])
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Average Vehicles")
    ax4.set_title("Weekly Traffic Curves")
    ax4.set_xticks(range(24))
    ax4.legend(facecolor=T["chart_bg"], labelcolor=T["text"], ncol=7)
    style_chart(fig4, ax4)
    show(fig4)

    st.subheader("🔍 Inspect a Single Day")
    sel_day = st.selectbox("Select a day", DAYS)
    sel = d[d["day"] == sel_day].groupby("hour")["vehicles"].mean()
    s1, s2, s3 = st.columns(3)
    card("Average", f"{sel.mean():.0f}", "vehicles per hour", s1)
    card("Peak Hour", fmt_hour(sel.idxmax()), f"{sel.max():.0f} vehicles", s2)
    card("Quietest Hour", fmt_hour(sel.idxmin()), f"{sel.min():.0f} vehicles", s3)

    fig5, ax5 = plt.subplots(figsize=(12, 4))
    ax5.plot(sel.index, sel.values, marker="o", linewidth=2.5, color="#38bdf8")
    ax5.fill_between(sel.index, sel.values, alpha=0.2, color="#38bdf8")
    ax5.set_xlabel("Hour of Day")
    ax5.set_ylabel("Average Vehicles")
    ax5.set_title(f"{sel_day} Traffic in {city}")
    ax5.set_xticks(range(24))
    style_chart(fig5, ax5)
    show(fig5)

    st.subheader("📋 Day-wise Summary")
    summary_days = pd.DataFrame({
        "Avg Vehicles": day_avg.round(0),
        "Avg Speed (km/h)": day_speed.round(1)
    })
    st.dataframe(summary_days)

# ---------------------------------------------------------------
# City comparison page
# ---------------------------------------------------------------
elif page == "🏙️ City Comparison":
    st.subheader("🏙️ City Comparison")

    city_names = list(CITIES.keys())
    city_avg = df.groupby("city")["vehicles"].mean().reindex(city_names)
    city_speed = df.groupby("city")["speed"].mean().reindex(city_names)

    cols = st.columns(5)
    for i, cname in enumerate(city_names):
        card(cname, f"{city_avg[cname]:.0f}", "avg vehicles per hour", cols[i])

    left, right = st.columns(2)

    with left:
        st.subheader("Average Vehicles per City")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(city_names, city_avg.values, color=CITY_COLORS)
        ax.set_ylabel("Average Vehicles")
        ax.set_title("Traffic Volume")
        style_chart(fig, ax)
        show(fig)

    with right:
        st.subheader("Average Speed per City")
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.bar(city_names, city_speed.values, color=CITY_COLORS)
        ax2.set_ylabel("Speed (km/h)")
        ax2.set_title("Average Speed")
        style_chart(fig2, ax2)
        show(fig2)

    st.subheader("📈 Hourly Traffic of All Cities")
    hc = df.groupby(["hour", "city"])["vehicles"].mean().unstack()
    fig3, ax3 = plt.subplots(figsize=(12, 4.5))
    for i, cname in enumerate(city_names):
        ax3.plot(hc.index, hc[cname], linewidth=2.5,
                 color=CITY_COLORS[i], label=cname)
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Average Vehicles")
    ax3.set_title("City-wise Hourly Traffic")
    ax3.set_xticks(range(24))
    ax3.legend(facecolor=T["chart_bg"], labelcolor=T["text"], ncol=5)
    style_chart(fig3, ax3)
    show(fig3)

    st.subheader("🚦 Congestion Share per City")
    share = pd.crosstab(df["city"], df["level"], normalize="index")
    share = share.reindex(index=city_names, columns=LEVELS).fillna(0) * 100
    fig4, ax4 = plt.subplots(figsize=(12, 4.5))
    bottom = np.zeros(len(city_names))
    for lvl in LEVELS:
        ax4.bar(city_names, share[lvl].values, bottom=bottom,
                color=LEVEL_COLORS[lvl], label=lvl)
        bottom = bottom + share[lvl].values
    ax4.set_ylabel("Share of Hours (%)")
    ax4.set_title("Low, Medium and High Traffic")
    ax4.legend(facecolor=T["chart_bg"], labelcolor=T["text"])
    style_chart(fig4, ax4)
    show(fig4)

    st.subheader("📋 City Summary")
    summary = df.groupby("city").agg(
        avg_vehicles=("vehicles", "mean"),
        peak_vehicles=("vehicles", "max"),
        avg_speed=("speed", "mean")
    ).round(1).reindex(city_names)
    summary["high_traffic_percent"] = (
        df["level"].eq("High").groupby(df["city"]).mean() * 100
    ).round(1)
    st.dataframe(summary)

# ---------------------------------------------------------------
# Prediction page
# ---------------------------------------------------------------
elif page == "🔮 Traffic Prediction":
    st.subheader(f"🔮 Predict Traffic Flow in {city}")

    now = now_ist()
    today = DAYS[now.weekday()]
    live = st.checkbox("⏱️ Use the current day and time (live)", value=False)

    left, right = st.columns(2)
    with left:
        sel_day = st.selectbox("📅 Day", DAYS, index=DAYS.index(today),
                               disabled=live)
        p_weather = st.selectbox("🌦️ Weather", WEATHER)
    with right:
        sel_hour = st.slider("🕐 Hour of Day", 0, 23, int(now.hour),
                             disabled=live)
        st.write("")
        predict_clicked = st.button("🔮 Predict Traffic")

    p_day = today if live else sel_day
    p_hour = int(now.hour) if live else int(sel_hour)

    if predict_clicked:
        st.session_state.last_pred = {
            "city": city, "day": p_day, "weather": p_weather, "hour": p_hour
        }

    lp = st.session_state.get("last_pred")
    if lp:
        sc = CITIES[lp["city"]]
        h = lp["hour"]
        v_all, s_all = predict_day(lp["city"], lp["day"], lp["weather"])
        pred_vehicles = int(round(v_all[h]))
        pred_speed = float(s_all[h])
        pred_level = level_of(v_all[h], sc)

        st.subheader(f"📌 Prediction Result: {lp['city']}")
        c1, c2, c3 = st.columns(3)
        card("Predicted Vehicles", f"{pred_vehicles}", "per hour", c1)
        card("Predicted Speed", f"{pred_speed:.1f} km/h", "average", c2)
        card("Traffic Level", f"{LEVEL_EMOJI[pred_level]} {pred_level}",
             f"{fmt_hour(h)} on {lp['day']} ({lp['weather']})", c3)

        if pred_level == "High":
            box("🚨 Heavy traffic expected. Avoid this hour if possible.", "#ef4444")
        elif pred_level == "Medium":
            box("⚠️ Moderate traffic expected. Allow some extra travel time.", "#facc15")
        else:
            box("✅ Light traffic expected. Roads should be clear.", "#22c55e")

        day_levels = [level_of(x, sc) for x in v_all]
        all_hours = list(range(24))

        st.subheader(f"📈 Full Day Forecast: {lp['day']} ({lp['weather']})")
        left2, right2 = st.columns(2)

        with left2:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(all_hours, v_all, color=[LEVEL_COLORS[x] for x in day_levels])
            ax.axvline(h, color=T["text"], linestyle="--", linewidth=1.5)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Vehicles")
            ax.set_title("Predicted Vehicles (dashed line = selected hour)")
            style_chart(fig, ax)
            show(fig)

        with right2:
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            ax2.plot(all_hours, s_all, marker="o", linewidth=2.5, color="#f472b6")
            ax2.axvline(h, color=T["text"], linestyle="--", linewidth=1.5)
            ax2.set_xlabel("Hour of Day")
            ax2.set_ylabel("Predicted Speed (km/h)")
            ax2.set_title("Predicted Speed")
            style_chart(fig2, ax2)
            show(fig2)

        day_series = pd.Series(v_all, index=all_hours)
        travel_hours = day_series.loc[6:22]
        best_hour = int(travel_hours.idxmin())
        worst_hour = int(day_series.idxmax())
        box(f"🟢 Best time to travel (6 AM to 10 PM): {fmt_hour(best_hour)} "
            f"with about {travel_hours.min():.0f} vehicles", "#22c55e")
        box(f"🔴 Busiest hour of the day: {fmt_hour(worst_hour)} "
            f"with about {day_series.max():.0f} vehicles", "#ef4444")

        st.subheader("📋 Hourly Forecast Table")
        forecast = pd.DataFrame({
            "Hour": [fmt_hour(x) for x in all_hours],
            "Vehicles": np.round(v_all).astype(int),
            "Speed (km/h)": np.round(s_all, 1),
            "Level": [f"{LEVEL_EMOJI[x]} {x}" for x in day_levels]
        })
        st.dataframe(forecast, hide_index=True)

        st.download_button(
            "⬇️ Download Forecast Report (CSV)",
            data=forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{lp['city']}_{lp['day']}.csv",
            mime="text/csv"
        )

        st.subheader("📤 Share This Prediction")
        share_msg = (
            f"🚦 Traffic forecast for {lp['city']} ({lp['day']}, {fmt_hour(h)}, "
            f"{lp['weather']} weather): about {pred_vehicles} vehicles per hour, "
            f"{pred_speed:.0f} km/h average speed, {pred_level} traffic."
        )
        show_links(share_buttons_html(share_msg, st.session_state.get("share_url", "")))
    else:
        box("👆 Choose a day, weather and hour, then click "
            "<b>Predict Traffic</b>.", "#6366f1")

# ---------------------------------------------------------------
# Compare scenarios page
# ---------------------------------------------------------------
elif page == "⚖️ Compare Scenarios":
    st.subheader("⚖️ Compare Two Scenarios")
    st.write("See how traffic changes with the city, day, weather or hour.")

    now = now_ist()
    today = DAYS[now.weekday()]
    city_list = list(CITIES.keys())

    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### 🅰️ Scenario A")
        a_city = st.selectbox("City (A)", city_list,
                              index=city_list.index(city), key="cmp_a_city")
        a_day = st.selectbox("Day (A)", DAYS,
                             index=DAYS.index(today), key="cmp_a_day")
        a_weather = st.selectbox("Weather (A)", WEATHER, index=0,
                                 key="cmp_a_weather")
        a_hour = st.slider("Hour (A)", 0, 23, 9, key="cmp_a_hour")
    with cb:
        st.markdown("#### 🅱️ Scenario B")
        b_city = st.selectbox("City (B)", city_list,
                              index=city_list.index(city), key="cmp_b_city")
        b_day = st.selectbox("Day (B)", DAYS,
                             index=DAYS.index(today), key="cmp_b_day")
        b_weather = st.selectbox("Weather (B)", WEATHER, index=2,
                                 key="cmp_b_weather")
        b_hour = st.slider("Hour (B)", 0, 23, 9, key="cmp_b_hour")

    va, sa = predict_day(a_city, a_day, a_weather)
    vb, sb = predict_day(b_city, b_day, b_weather)
    a_v, a_s = float(va[a_hour]), float(sa[a_hour])
    b_v, b_s = float(vb[b_hour]), float(sb[b_hour])
    a_lvl = level_of(a_v, CITIES[a_city])
    b_lvl = level_of(b_v, CITIES[b_city])

    st.subheader("📌 Results")
    r1 = st.columns(3)
    card("🅰️ Vehicles", f"{a_v:.0f}", f"{a_city}, {a_day[:3]} {fmt_hour(a_hour)}", r1[0])
    card("🅰️ Speed", f"{a_s:.1f} km/h", a_weather, r1[1])
    card("🅰️ Level", f"{LEVEL_EMOJI[a_lvl]} {a_lvl}", "traffic", r1[2])
    r2 = st.columns(3)
    card("🅱️ Vehicles", f"{b_v:.0f}", f"{b_city}, {b_day[:3]} {fmt_hour(b_hour)}", r2[0])
    card("🅱️ Speed", f"{b_s:.1f} km/h", b_weather, r2[1])
    card("🅱️ Level", f"{LEVEL_EMOJI[b_lvl]} {b_lvl}", "traffic", r2[2])

    if a_v > 0:
        pct = (b_v - a_v) / a_v * 100
        more_less = "more" if pct >= 0 else "fewer"
        box(f"Scenario B has about <b>{abs(pct):.0f}% {more_less}</b> vehicles "
            f"than Scenario A.", "#6366f1")
    speed_diff = b_s - a_s
    box(f"Average speed in B is <b>{abs(speed_diff):.1f} km/h "
        f"{'faster' if speed_diff >= 0 else 'slower'}</b> than in A.", "#06b6d4")

    st.subheader("📈 24 Hour Comparison")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(range(24), va, linewidth=2.5, color="#38bdf8",
            label=f"A: {a_city}, {a_day[:3]}, {a_weather}")
    ax.plot(range(24), vb, linewidth=2.5, color="#f472b6",
            label=f"B: {b_city}, {b_day[:3]}, {b_weather}")
    ax.scatter([a_hour], [a_v], color="#38bdf8", s=70, zorder=5)
    ax.scatter([b_hour], [b_v], color="#f472b6", s=70, zorder=5)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Vehicles")
    ax.set_title("Scenario A vs Scenario B")
    ax.set_xticks(range(24))
    ax.legend(facecolor=T["chart_bg"], labelcolor=T["text"])
    style_chart(fig, ax)
    show(fig)

# ---------------------------------------------------------------
# Trip planner page
# ---------------------------------------------------------------
elif page == "🧭 Trip Planner":
    st.subheader(f"🧭 Smart Trip Planner: {city}")
    st.write("Find the best time to leave, then open the route in Google Maps "
             "to see the real live traffic.")

    now = now_ist()
    today = DAYS[now.weekday()]

    t1, t2 = st.columns(2)
    with t1:
        origin = st.text_input("📍 From", placeholder="For example: Kothrud")
        distance = st.number_input("📏 Trip distance (km)", min_value=1.0,
                                   max_value=300.0, value=10.0, step=1.0)
        t_day = st.selectbox("📅 Day of travel", DAYS, index=DAYS.index(today))
    with t2:
        dest = st.text_input("🏁 To", placeholder="For example: Hinjewadi")
        t_weather = st.selectbox("🌦️ Weather", WEATHER)
        window = st.slider("🕐 Departure window (hours)", 0, 23, (7, 11))

    hours = list(range(window[0], window[1] + 1))
    v_w, s_w = predict_day(city, t_day, t_weather, hours)
    minutes = float(distance) / s_w * 60
    lv_w = [level_of(x, scale) for x in v_w]
    best_i = int(np.argmin(minutes))
    worst_i = int(np.argmax(minutes))

    st.subheader("✅ Best Time to Leave")
    c1, c2, c3, c4 = st.columns(4)
    card("Best Departure", fmt_hour(hours[best_i]), t_day, c1)
    card("Travel Time", f"{minutes[best_i]:.0f} min", f"for {distance:.0f} km", c2)
    card("Average Speed", f"{s_w[best_i]:.0f} km/h", f"{LEVEL_EMOJI[lv_w[best_i]]} {lv_w[best_i]} traffic", c3)
    card("Time Saved", f"{minutes[worst_i] - minutes[best_i]:.0f} min",
         f"vs leaving at {fmt_hour(hours[worst_i])}", c4)

    v_n, s_n = predict_day(city, today, t_weather, [now.hour])
    min_now = float(distance) / float(s_n[0]) * 60
    lvl_n = level_of(v_n[0], scale)
    box(f"⏱️ If you leave <b>right now</b> ({now.strftime('%H:%M')} IST, {today}): "
        f"about {min_now:.0f} min for {distance:.0f} km, "
        f"{LEVEL_EMOJI[lvl_n]} {lvl_n} traffic.", LEVEL_COLORS[lvl_n])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(hours, minutes, color=[LEVEL_COLORS[x] for x in lv_w])
    ax.set_xticks(hours)
    ax.set_xlabel("Departure Hour")
    ax.set_ylabel("Travel Time (minutes)")
    ax.set_title("Estimated Travel Time by Departure Hour")
    style_chart(fig, ax)
    show(fig)

    st.subheader("📋 Departure Options")
    options = pd.DataFrame({
        "Leave at": [fmt_hour(x) for x in hours],
        "Vehicles": np.round(v_w).astype(int),
        "Speed (km/h)": np.round(s_w, 1),
        "Travel time (min)": np.round(minutes, 0).astype(int),
        "Level": [f"{LEVEL_EMOJI[x]} {x}" for x in lv_w],
    })
    st.dataframe(options, hide_index=True)

    st.subheader("🗺️ Open in Google Maps (real live traffic)")
    buttons = []
    if origin.strip() and dest.strip():
        route_url = ("https://www.google.com/maps/dir/?api=1&origin="
                     + quote(origin.strip() + ", " + city)
                     + "&destination=" + quote(dest.strip() + ", " + city)
                     + "&travelmode=driving")
        buttons.append(link_btn("🧭 Open Route", route_url, "#16a34a"))
    buttons.append(link_btn(f"🚦 Live Traffic in {city}", live_traffic_url(city), "#2563eb"))
    show_links("<div class='linkrow'>" + "".join(buttons) + "</div>")
    if not (origin.strip() and dest.strip()):
        st.caption("Enter both From and To to get the Open Route button.")

    st.subheader("📤 Share This Plan")
    place = f"{origin.strip()} to {dest.strip()}" if origin.strip() and dest.strip() else "my trip"
    plan_msg = (f"🚦 Best time to leave for {place} in {city} on {t_day}: "
                f"{fmt_hour(hours[best_i])} (about {minutes[best_i]:.0f} min).")
    show_links(share_buttons_html(plan_msg, st.session_state.get("share_url", "")))

    st.caption("Travel time = distance divided by the predicted average speed. "
               "It is an estimate from the model, not live GPS data.")

# ---------------------------------------------------------------
# Traffic map page
# ---------------------------------------------------------------
elif page == "🗺️ Traffic Map":
    st.subheader(f"🗺️ Traffic Map: {city}")
    st.write("Green areas are free, yellow are moderate and red are jammed. "
             "Tap or hover over a dot for details.")

    now = now_ist()
    today = DAYS[now.weekday()]
    live_m = st.checkbox("⏱️ Use the current day and time (live)", value=True,
                         key="map_live")
    m1, m2, m3 = st.columns(3)
    with m1:
        m_day_sel = st.selectbox("📅 Day", DAYS, index=DAYS.index(today),
                                 disabled=live_m, key="map_day")
    with m2:
        m_hour_sel = st.slider("🕐 Hour", 0, 23, int(now.hour),
                               disabled=live_m, key="map_hour")
    with m3:
        m_weather = st.selectbox("🌦️ Weather", WEATHER, key="map_weather")

    m_day = today if live_m else m_day_sel
    m_hour = int(now.hour) if live_m else int(m_hour_sel)

    base_v, base_s = predict_day(city, m_day, m_weather, [m_hour])
    rows = []
    for name, lat, lon, factor in ROADS[city]:
        v = float(base_v[0]) * factor
        sp = float(np.clip(base_s[0] - (factor - 1.0) * 15, 8, 60))
        lvl = level_of(v, scale)
        rows.append({
            "name": name, "lat": lat, "lon": lon, "level": lvl,
            "vehicles": int(round(v)), "speed": round(sp, 1),
            "color": LEVEL_RGB[lvl],
        })
    map_df = pd.DataFrame(rows)

    c_lat, c_lon, c_zoom = CITY_VIEW[city]
    show_map(map_df, c_lat, c_lon, c_zoom)

    ranked = map_df.sort_values("vehicles").reset_index(drop=True)
    best3 = ", ".join(ranked["name"].head(3))
    worst3 = ", ".join(ranked["name"].tail(3)[::-1])
    box(f"✅ <b>Least crowded now:</b> {best3}", "#22c55e")
    box(f"⛔ <b>Most crowded now:</b> {worst3}", "#ef4444")

    st.subheader("📋 All Areas (least crowded first)")
    table = pd.DataFrame({
        "Area / Road": ranked["name"],
        "Level": [f"{LEVEL_EMOJI[x]} {x}" for x in ranked["level"]],
        "Vehicles per hour": ranked["vehicles"],
        "Speed (km/h)": ranked["speed"],
    })
    st.dataframe(table, hide_index=True)

    st.subheader("🌐 Want the real live road traffic?")
    show_links("<div class='linkrow'>"
               + link_btn(f"🚦 Open Live Traffic in {city}", live_traffic_url(city), "#2563eb")
               + "</div>")
    st.caption("Area positions are approximate and the busy level of each "
               "area is estimated from the model, not from live GPS. "
               "Use the Google Maps button above for real-time road conditions.")

# ---------------------------------------------------------------
# Model accuracy page
# ---------------------------------------------------------------
elif page == "🎯 Model Accuracy":
    st.subheader("🎯 How Accurate Are the Models?")
    res = evaluate_models()

    box(f"The data is split into <b>{res['train_rows']:,}</b> rows for training "
        f"and <b>{res['test_rows']:,}</b> unseen rows for testing.", "#6366f1")

    c1, c2, c3 = st.columns(3)
    card("Vehicle Model R²", f"{res['v_r2']:.3f}", "1.0 is perfect", c1)
    card("Vehicle Error (MAE)", f"{res['v_mae']:.1f}", "vehicles per hour", c2)
    card("Traffic Level Accuracy", f"{res['level_acc']:.1f}%", "Low / Medium / High", c3)
    c4, c5, _ = st.columns(3)
    card("Speed Model R²", f"{res['s_r2']:.3f}", "1.0 is perfect", c4)
    card("Speed Error (MAE)", f"{res['s_mae']:.2f}", "km/h", c5)

    left, right = st.columns(2)
    with left:
        st.subheader("Actual vs Predicted Vehicles")
        n_pts = min(600, len(res["v_true"]))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(res["v_true"][:n_pts], res["v_pred"][:n_pts],
                   s=12, alpha=0.6, color="#38bdf8")
        top = float(max(res["v_true"].max(), res["v_pred"].max()))
        ax.plot([0, top], [0, top], linestyle="--", color="#f472b6")
        ax.set_xlabel("Actual Vehicles")
        ax.set_ylabel("Predicted Vehicles")
        ax.set_title("Points near the dashed line = good predictions")
        style_chart(fig, ax)
        show(fig)

    with right:
        st.subheader("Feature Importance")
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.bar(FEATURE_LABELS, res["importance"] * 100,
                color=["#38bdf8", "#a78bfa", "#fb923c", "#4ade80"])
        ax2.set_ylabel("Importance (%)")
        ax2.set_title("What affects vehicle count the most")
        style_chart(fig2, ax2)
        show(fig2)

    box("📘 <b>R²</b> shows how much of the traffic pattern the model explains. "
        "<b>MAE</b> is the average mistake in the prediction.", "#06b6d4")
    box("⚠️ The dataset is generated by a program with a clear pattern, so scores "
        "are very high. Real traffic data is noisier, so scores would be lower.", "#f59e0b")

# ---------------------------------------------------------------
# Upload data page
# ---------------------------------------------------------------
elif page == "📤 Upload Data":
    st.subheader("📤 Use Your Own Real Traffic Data")
    box("Upload a CSV file with these columns: <b>city, day, hour, weather, "
        "vehicles, speed</b>. City must be one of Pune, Mumbai, Nagpur, Nashik "
        "or Delhi. Day can be Monday or Mon. Weather must be Clear, Cloudy or "
        "Rainy. Hour is 0 to 23. At least 50 rows are needed.", "#6366f1")

    st.download_button(
        "⬇️ Download CSV Template",
        data=df[UPLOAD_COLS].head(48).to_csv(index=False).encode("utf-8"),
        file_name="traffic_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        raw = None
        try:
            raw = pd.read_csv(uploaded)
        except Exception:
            box("Could not read this file. Please upload a valid CSV.", "#ef4444")
        if raw is not None:
            clean, message = clean_uploaded(raw)
            if clean is None:
                box(message, "#ef4444")
            else:
                box(f"✅ {len(clean):,} valid rows found out of {len(raw):,} "
                    f"rows in your file.", "#22c55e")
                st.dataframe(clean.head(10), hide_index=True)
                if st.button("🚀 Train Model on My Data"):
                    with st.spinner("Training your model..."):
                        st.session_state.custom_models = train_custom(clean)
                        st.session_state.custom_rows = int(len(clean))
                        st.session_state.use_custom = True
                    st.rerun()

    if st.session_state.get("use_custom") and st.session_state.get("custom_models"):
        box(f"🧪 Your model (trained on {st.session_state.get('custom_rows', 0):,} rows) "
            f"is active in Prediction, Compare, Trip Planner, Map and Home.", "#22c55e")
        if st.button("↩️ Switch Back to the Default Model"):
            st.session_state.use_custom = False
            st.rerun()
    st.caption("Include data for every city you want to predict. "
               "Charts on the Dashboard pages still show the built-in dataset.")

# ---------------------------------------------------------------
# Dataset page
# ---------------------------------------------------------------
elif page == "🗂️ Dataset":
    st.subheader("🗂️ Traffic Dataset")

    f1, f2, f3 = st.columns(3)
    with f1:
        sel_cities = st.multiselect("🏙️ City", list(CITIES.keys()),
                                    default=[city])
    with f2:
        sel_days = st.multiselect("📅 Day", DAYS, default=DAYS)
    with f3:
        sel_levels = st.multiselect("🚦 Traffic Level", LEVELS, default=LEVELS)

    show_cols = ["city", "week", "day", "hour", "weather",
                 "vehicles", "speed", "level"]
    view = df[
        df["city"].isin(sel_cities)
        & df["day"].isin(sel_days)
        & df["level"].isin(sel_levels)
    ][show_cols]

    st.write(f"Showing **{len(view):,}** of **{len(df):,}** records")
    st.dataframe(view, hide_index=True)

    st.download_button(
        "⬇️ Download as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="traffic_data.csv",
        mime="text/csv"
    )

    st.subheader("📊 Statistics")
    st.dataframe(view[["vehicles", "speed"]].describe().round(1))

# ---------------------------------------------------------------
# Share page
# ---------------------------------------------------------------
elif page == "🔗 Share App":
    st.subheader("🔗 Share This App With Friends and Family")

    url_in = st.text_input(
        "🌐 Your app link",
        value=st.session_state.get("share_url", ""),
        placeholder="https://your-app.streamlit.app"
    ).strip()
    if url_in and not url_in.startswith(("http://", "https://")):
        box("The link must start with http:// or https://", "#ef4444")
        url_in = ""
    st.session_state.share_url = url_in

    intro = st.text_area(
        "✏️ Message",
        value=("🚦 I'm using the Traffic Flow Prediction app to check traffic "
               "and find the best time to travel in Pune, Mumbai, Nagpur, "
               "Nashik and Delhi. Try it!"),
        height=100
    )

    if url_in:
        st.subheader("📤 Send")
        show_links(share_buttons_html(intro, url_in))
        st.write("Copy the message:")
        st.code((intro + " " + url_in).strip(), language=None)
        st.subheader("📱 Scan to Open")
        qr_url = ("https://api.qrserver.com/v1/create-qr-code/?size=220x220&data="
                  + quote(url_in, safe=""))
        st.markdown(
            f"<img src='{html.escape(qr_url, quote=True)}' width='180' "
            "style='background:#ffffff; padding:8px; border-radius:12px;'>",
            unsafe_allow_html=True
        )
        st.caption("The QR image is made by the free service api.qrserver.com.")
    else:
        box("Paste your app link above to get the share buttons and QR code. "
            "You can still send the message below on WhatsApp or Email.", "#f59e0b")
        show_links(share_buttons_html(intro, ""))

    st.subheader("🌍 How to Get a Link to Share")
    box("<b>Option 1: Same Wi-Fi.</b> Run <code>streamlit run app.py</code>. "
        "The terminal shows a <b>Network URL</b>. Anyone on the same Wi-Fi "
        "can open it.", "#06b6d4")
    box("<b>Option 2: Public link (free).</b> Upload <code>app.py</code> and "
        "<code>requirements.txt</code> to a GitHub repository, then deploy it "
        "at share.streamlit.io. You get a link like "
        "<code>https://yourname-traffic.streamlit.app</code> that works "
        "anywhere.", "#22c55e")
    box("⚠️ On free hosting the <code>users.json</code> file can reset when the "
        "app restarts, so accounts may be lost. For permanent accounts a "
        "database is needed.", "#f59e0b")
    st.write("Your <b>requirements.txt</b> should contain:", unsafe_allow_html=True)
    st.code("streamlit\npandas\nnumpy\nmatplotlib\nscikit-learn\npydeck", language=None)

# ---------------------------------------------------------------
# About page
# ---------------------------------------------------------------
elif page == "ℹ️ About":
    st.subheader("ℹ️ About This Project")
    st.write(
        "The Traffic Flow Prediction System analyses and predicts traffic "
        "for five major Indian cities using machine learning."
    )

    box("🎯 <b>Objective</b>: predict the number of vehicles and the average "
        "speed for any city, day, hour and weather, and help people choose "
        "the best time to travel.", "#6366f1")
    box("🧠 <b>Model</b>: two Random Forest Regressors, one for vehicle count "
        "and one for speed.", "#06b6d4")
    box("📥 <b>Input features</b>: city, day of the week, hour of the day "
        "and weather.", "#a855f7")
    box("🗃️ <b>Dataset</b>: simulated hourly data for 8 weeks across 5 cities, "
        "with rush hour patterns, weekday and weekend differences and "
        "weather effects. Users can also upload their own real data.", "#22c55e")
    box("🚦 <b>Traffic levels</b>: Low, Medium and High, based on vehicles per "
        "hour adjusted for city size.", "#f59e0b")
    box("👤 <b>Accounts</b>: users can create an account. Passwords are stored "
        "as salted hashes, never as plain text.", "#ec4899")
    box("🎨 <b>Themes</b>: Dark, Light, Ocean and Sunset, saved per account.", "#14b8a6")

    st.subheader("🛠️ Technologies Used")
    t1, t2, t3, t4 = st.columns(4)
    card("Streamlit", "UI", "web app framework", t1)
    card("Pandas / NumPy", "Data", "processing and generation", t2)
    card("Scikit-learn", "ML", "Random Forest models", t3)
    card("Matplotlib / PyDeck", "Charts", "graphs and map", t4)

    st.subheader("⚠️ Note")
    st.write(
        "The built-in data is generated by a program, not collected from real "
        "sensors, so predictions show how the system works and are not real "
        "traffic forecasts. For real road conditions, use the Google Maps "
        "buttons in the Trip Planner and Traffic Map pages."
    )