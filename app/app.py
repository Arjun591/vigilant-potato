import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import streamlit.components.v1 as components
import os

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Crime Hotspot Detector",
    page_icon="🚨",
    layout="wide"
)

# ─── Load Data ─────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "crime_data_cleaned.csv"))
    city_summary = pd.read_csv(os.path.join(BASE_DIR, "data", "city_summary.csv"))
    hotspot_zones = pd.read_csv(os.path.join(BASE_DIR, "data", "hotspot_zones.csv"))
    return df, city_summary, hotspot_zones

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "model", "saved_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model", "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

df, city_summary, hotspot_zones = load_data()
model, encoders = load_model()

# ─── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/rotating-light-emoji.png", width=80)
st.sidebar.title("🚨 Crime Hotspot Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Crime Analysis",
    "🎯 Victim & Weapon Analysis",
    "🗺️ Hotspot Map",
    "🤖 ML Predictions",
    "🔮 Live Crime Predictor",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Cities Covered", df["City"].nunique() if "City" in df.columns else "N/A")

# ─── Helper ────────────────────────────────────────────────────
def show_chart(filename, caption=""):
    path = os.path.join(BASE_DIR, "visualizations", filename)
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption, use_column_width=True)
    else:
        st.warning(f"Chart not found: {filename}")

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🚨 AI Crime Hotspot Detector")
    st.markdown("### Predicting and Visualizing Crime Patterns Using Machine Learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Crimes", f"{len(df):,}")
    col2.metric("Cities", df["City"].nunique() if "City" in df.columns else "N/A")
    col3.metric("Crime Types", df["Crime Description"].nunique() if "Crime Description" in df.columns else "N/A")
    col4.metric("Case Closed %",
        f"{(df['Case Closed'].str.upper() == 'YES').mean()*100:.1f}%" 
        if "Case Closed" in df.columns else "N/A"
    )

    st.markdown("---")
    st.subheader("📌 Project Summary")
    st.markdown("""
    This dashboard uses **Machine Learning** to detect crime hotspots and analyze patterns across cities.

    **What this app does:**
    - 📊 Analyzes crime trends by city, time, and type
    - 🎯 Profiles victims and weapon usage
    - 🗺️ Maps crime hotspot zones
    - 🤖 Predicts crime clusters using KMeans + Random Forest
    """)

    st.markdown("---")
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — Crime Analysis
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Crime Analysis":
    st.title("📊 Crime Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["By City", "By Domain", "By Time", "By Month/Day"])

    with tab1:
        st.subheader("Crimes by City")
        show_chart("chart1_crimes_by_city.png")
        if not city_summary.empty:
            st.dataframe(city_summary, use_container_width=True)

    with tab2:
        st.subheader("Crimes by Domain")
        show_chart("chart2_crimes_by_domain.png")

    with tab3:
        st.subheader("Crimes by Hour of Day")
        show_chart("chart3_crimes_by_hour.png")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("By Month")
            show_chart("chart4_crimes_by_month.png")
        with col2:
            st.subheader("By Day of Week")
            show_chart("chart5_crimes_by_day.png")

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — Victim & Weapon Analysis
# ═══════════════════════════════════════════════════════════════
elif page == "🎯 Victim & Weapon Analysis":
    st.title("🎯 Victim & Weapon Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Victim Gender", "Weapons Used", "State Trends"])

    with tab1:
        st.subheader("Victim Gender Distribution")
        show_chart("chart7_victim_gender.png")

        if "Victim Gender" in df.columns:
            gender_counts = df["Victim Gender"].value_counts().reset_index()
            gender_counts.columns = ["Gender", "Count"]
            st.dataframe(gender_counts, use_container_width=True)

    with tab2:
        st.subheader("Weapons Used in Crimes")
        show_chart("chart6_weapons.png")

        if "Weapon Used" in df.columns:
            weapon_counts = df["Weapon Used"].value_counts().head(10).reset_index()
            weapon_counts.columns = ["Weapon", "Count"]
            st.dataframe(weapon_counts, use_container_width=True)

    with tab3:
        st.subheader("State-wise Crime Trends")
        show_chart("chart12_state_trends.png")

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — Hotspot Map
# ═══════════════════════════════════════════════════════════════
elif page == "🗺️ Hotspot Map":
    st.title("🗺️ Crime Hotspot Map")
    st.markdown("---")

    map_path = os.path.join(BASE_DIR, "visualizations", "crime_hotspot_map.html")
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()
        components.html(map_html, height=600, scrolling=True)
    else:
        st.error("Map file not found. Make sure crime_hotspot_map.html is in the visualizations folder.")

    st.markdown("---")
    st.subheader("📍 Hotspot Zones Table")
    st.dataframe(hotspot_zones, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — ML Predictions
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 ML Predictions":
    st.title("🤖 ML Predictions")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Clustering", "Model Performance", "Feature Importance"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Elbow Curve (Optimal K)")
            show_chart("elbow_curve.png")
        with col2:
            st.subheader("Crime Clusters")
            show_chart("chart8_clusters.png")

        st.subheader("Hotspot Zones")
        show_chart("chart9_hotspot_zones.png")

    with tab2:
        st.subheader("Confusion Matrix")
        show_chart("chart11_confusion_matrix.png")

    with tab3:
        st.subheader("Feature Importance")
        show_chart("chart10_feature_importance.png")

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — Live Crime Predictor (Fixed Version)
# ═══════════════════════════════════════════════════════════════
elif page == "🔮 Live Crime Predictor":
    st.title("🔮 Live Crime Predictor")
    st.markdown("Enter details to predict the most likely crime domain.")
    st.markdown("---")

    # This helper function handles the naming mismatch automatically
    def get_encoder(search_key):
        # Checks if the exact key exists, otherwise tries to find a partial match
        if search_key in encoders:
            return encoders[search_key]
        for k in encoders.keys():
            if search_key.split('_')[-1] in k.lower():
                return encoders[k]
        return None

    def get_time_of_day(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 21: return 'Evening'
        else: return 'Night'

    # Mapping your UI to the encoders we know exist in your model
    enc_city = get_encoder('city')
    enc_weapon = get_encoder('weapon')
    enc_gender = get_encoder('gender')
    enc_target = get_encoder('target')
    enc_day = get_encoder('day')
    enc_tod = get_encoder('tod')

    if not enc_city or not enc_weapon:
        st.error("Could not find matching encoders in model/encoders.pkl.")
        st.write("Current keys in your file:", list(encoders.keys()))
    else:
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("🏙️ City", sorted(enc_city.classes_))
            hour = st.slider("🕐 Hour of Day", 0, 23, 12)
            day = st.selectbox("📅 Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        with col2:
            weapon = st.selectbox("🔪 Weapon Used", sorted(enc_weapon.classes_))
            age = st.slider("👤 Victim Age", 1, 90, 25)
            gender = st.selectbox("⚧ Victim Gender", enc_gender.classes_ if enc_gender else ["M", "F", "X"])
        
        month = st.slider("📆 Month", 1, 12, 6)
        is_closed = st.radio("📁 Case Status", ["Open", "Closed"], horizontal=True)

        if st.button("🔮 Predict Crime Domain", use_container_width=True):
            try:
                tod = get_time_of_day(hour)
                
                # Use the encoders found above
                row = pd.DataFrame({
                    'City_enc':   [enc_city.transform([city])[0]],
                    'Hour':       [hour],
                    'Day_enc':    [enc_day.transform([day])[0] if enc_day else 0],
                    'Weapon_enc': [enc_weapon.transform([weapon])[0]],
                    'Victim Age': [age],
                    'ToD_enc':    [enc_tod.transform([tod])[0] if enc_tod else 0],
                    'Month_Num':  [month],
                    'Is Closed':  [1 if is_closed == "Closed" else 0],
                    'Gender_enc': [enc_gender.transform([gender])[0] if enc_gender else 0]
                })

                prediction = model.predict(row)[0]
                probabilities = model.predict_proba(row)[0]
                predicted_label = enc_target.classes_[prediction]

                st.success(f"### 🎯 Predicted Domain: **{predicted_label}**")
                
                prob_df = pd.DataFrame({
                    'Domain': enc_target.classes_,
                    'Probability': [p * 100 for p in probabilities]
                }).sort_values('Probability', ascending=False)
                
                for _, p_row in prob_df.iterrows():
                    st.write(f"**{p_row['Domain']}** ({p_row['Probability']:.1f}%)")
                    st.progress(int(p_row['Probability']))

            except Exception as e:
                st.error(f"Prediction logic error: {e}")