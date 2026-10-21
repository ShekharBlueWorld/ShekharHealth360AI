# Shekhar Health360 AI

Responsive Streamlit proof of concept combining nine months of illustrative CGM, Oura-style recovery, blood pressure and weight data. Quarterly A1c (8.0, 7.4, 7.1%) and October 20 weight (147 lb) are user-reported.

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0
```

Open `http://localhost:8501` on the Mac. For mobile, use the Network URL printed in Terminal on a phone connected to the same Wi-Fi. The Mac must stay awake. Network address can change. Public hosting is not included.

**Data disclosure:** All daily measurements, meal logs, and model performance are synthetic. Correlations and forecasts are for software demonstration only, not clinical use.
