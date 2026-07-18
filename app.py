# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import date

# -----------------------
# Page config & styling
# -----------------------
st.set_page_config(page_title="Crop Price Predictor", layout="wide", page_icon="🌾")

# Small CSS for a nicer look
st.markdown(
    """
    <style>
    .header {
        background: linear-gradient(90deg,#2b9af3,#38d39f);
        padding: 18px;
        border-radius: 12px;
        color: white;
        margin-bottom: 18px;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }
    .big-number {
        font-size: 34px;
        font-weight: 700;
        color: #0b5fff;
    }
    .muted {
        color: #6b7280;
        font-size: 13px;
    }
    .btn-space { margin-top: 8px }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="header"><h2 style="margin:0">🌾 Karnataka Crop Price Predictor</h2>'
            '<div class="muted">Enter weather & price inputs and click Predict</div></div>',
            unsafe_allow_html=True)

# -----------------------
# Load model (safe)
# -----------------------
MODEL_FILE = "crop_model.pkl"   # change if your model filename differs
model = None
try:
    model = joblib.load(MODEL_FILE)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(f"Model not found or failed to load: {MODEL_FILE}. Using a dummy fallback predictor. ({e})")

# -----------------------
# Left: Inputs panel
# -----------------------
with st.container():
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔎 Select Crop & Location")
        # if your dataset has known options, you can replace these lists with real ones
        district = st.selectbox("District / Market", ["Bengaluru Urban","Bengaluru Rural","Ramanagara","Chikkaballapura",
"Chitradurga","Davanagere","Kolar","Shivamogga","Tumakuru",
"Bagalkot","Ballari","Belagavi","Bidar","Bijapur","Dharwad",
"Gadag","Haveri","Kalaburagi","Koppal","Raichur",
"Vijayanagara","Yadgir","Mysuru","Mandya","Chamarajanagar",
"Udupi","Dakshina Kannada","Uttara Kannada","Kodagu","Hassan",
"Chikkamagaluru"], index=0)
        commodity = st.selectbox("Commodity", ["Rice"], index=0)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🗓 Date (for record)")
        dt = st.date_input("Pick a date", date.today())

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Prices (Rs/Quintal)")
        c1, c2 = st.columns(2)
        with c1:
            minimum = st.number_input("Minimum price", min_value=0.0, value=1000.0, step=10.0, format="%.2f")
        with c2:
            maximum = st.number_input("Maximum price", min_value=0.0, value=2000.0, step=10.0, format="%.2f")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🌦 Weather / Conditions (input)")
        r1, r2, r3 = st.columns([1,1,1])
        with r1:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0, step=1.0)
        with r2:
            temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=27.0, step=0.1, format="%.1f")
        with r3:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Middle / Right: Action + Output
# -----------------------
st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
out_col1, out_col2 = st.columns([2,3])

with out_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ▶️ Run Prediction")
    if st.button("Predict Price", key="predict_button"):
        # prepare features in the same order as your model expects
        # user code used [Y, M, D, minimum, maximum, temp, humidity, rainfall]
        Y = dt.year
        M = dt.month
        D = dt.day
        features = np.array([[Y, M, D, minimum, maximum, temp, humidity, rainfall]])
        try:
            if model_loaded:
                prediction = model.predict(features)[0]
            else:
                # fallback simple rule-based prediction: average of min & max slightly modified
                prediction = (minimum + maximum) / 2 * (1 + (0.02*(rainfall/1000)) - (0.01*((temp-25)/5)))
                prediction = float(prediction)
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            prediction = None

        # show big result card
        if prediction is not None:
            st.markdown(f'<div style="padding:12px;border-radius:8px;background:linear-gradient(90deg, #fff7ed, #fff1f2);">'
                        f'<div class="muted">Predicted modal price (Rs/Quintal)</div>'
                        f'<div class="big-number">₹{prediction:,.2f}</div></div>', unsafe_allow_html=True)
            # save to small dataframe for download
            pred_df = pd.DataFrame([{
                "Date": dt, "District": district, "Commodity": commodity,
                "Min_Price": minimum, "Max_Price": maximum,
                "Rainfall_mm": rainfall, "Temperature_C": temp, "Humidity_pct": humidity,
                "Predicted_Price": prediction
            }])
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download prediction CSV", data=csv, file_name="prediction.csv")
    else:
        st.info("Enter inputs and click Predict")

    st.markdown("</div>", unsafe_allow_html=True)

with out_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Quick visualization")
    # small timeseries plot using min, max and predicted value (toy)
    try:
        # create a tiny demonstration series: last 6 months synthetic + predicted
        last_date = pd.to_datetime(dt)
        months = pd.date_range(end=last_date, periods=6, freq='M')
        prices = np.linspace(minimum, maximum, num=6).tolist()
        # if we have a prediction, append it
        if 'prediction' in locals() and prediction is not None:
            months = months.append(pd.DatetimeIndex([last_date + pd.DateOffset(months=1)]))
            prices = prices + [prediction]
        series_df = pd.DataFrame({"Date": months, "Price": prices})
        # plot
        fig = None
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,2.4))
        ax.plot(series_df['Date'], series_df['Price'], marker='o')
        ax.fill_between(series_df['Date'], series_df['Price'], alpha=0.05)
        ax.set_title("Historical (sample) → next predicted")
        ax.set_ylabel("Price (₹/quintal)")
        ax.set_xlabel("")
        plt.xticks(rotation=25)
        st.pyplot(fig)
    except Exception as e:
        st.write("Visualization skipped:", e)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Footer: Model info & feature importance (if available)
# -----------------------
st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
with st.container():
    a, b = st.columns([3,1])
    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
