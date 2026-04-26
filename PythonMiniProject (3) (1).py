import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("📈 Stock Price Analysis & Prediction App")


# SIDEBAR USER INPUTS

st.sidebar.header("User Input Options")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")

# Quick preset time ranges
st.sidebar.subheader("Quick Time Range")
preset = st.sidebar.selectbox(
    "Select Time Period",
    ["Custom Range", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "10 Years"]
)

# Compute date ranges
today = datetime.today()

if preset == "Custom Range":
    start = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
    end = st.sidebar.date_input("End Date", today)

elif preset == "1 Month":
    start = today - timedelta(days=30)
    end = today

elif preset == "3 Months":
    start = today - timedelta(days=90)
    end = today

elif preset == "6 Months":
    start = today - timedelta(days=182)
    end = today

elif preset == "1 Year":
    start = today - timedelta(days=365)
    end = today

elif preset == "5 Years":
    start = today - timedelta(days=5*365)
    end = today

elif preset == "10 Years":
    start = today - timedelta(days=10*365)
    end = today



 
# LOAD DATA

st.subheader(f"📥 Downloading Data for: {ticker}")

if preset == "Max":
    df = yf.download(ticker)
else:
    df = yf.download(ticker, start=start, end=end)

if df.empty:
    st.error("⚠️ No data found. Try a different ticker.")
    st.stop()
#Data Preproccesing
df = df.reset_index()
df = df.ffill()        # Fill forward missing values
df = df.bfill()        # Fill backward missing values
df = df.dropna()       # Remove any remaining null rows






# Fetch currency information from ticker
info = yf.Ticker(ticker).info
stock_currency = info.get("currency", "USD")

# Currency symbol mapping
currency_symbols = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "JPY": "¥",
    "GBP": "£",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "CHF",
    "CNY": "¥",
    "SGD": "S$",
    "HKD": "HK$"
}

# Select symbol
currency_symbol = currency_symbols.get(stock_currency, stock_currency)

st.write(f"**Detected Currency:** {stock_currency} ({currency_symbol})")

df_display=df.copy()

# Rename the columns directly 
df_display.rename(columns={
    "Open": f"Open ({currency_symbol})",
    "High": f"High ({currency_symbol})",
    "Low":  f"Low ({currency_symbol})",
    "Close": f"Close ({currency_symbol})"
}, inplace=True)

# Display updated dataframe with currency symbols

st.write("Data")
st.dataframe(df_display)







# PLOT 1 – DYNAMIC PRICE SELECTION

st.subheader("📌 Price Chart")

price_option = st.selectbox(
    "Select price type to plot:",
    ["Close", "Open", "High", "Low"]

)

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df[price_option])
plt.xlabel("Date")
plt.ylabel(f"{price_option} Price")
plt.title(f"{ticker} — {price_option} Price Over Time")
st.pyplot(plt)



# PLOT 2 – MOVING AVERAGES

st.subheader("📌 Moving Averages (100-day & 200-day)")

ma100 = df["Close"].rolling(100).mean()
ma200 = df["Close"].rolling(200).mean()

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Close")
plt.plot(df["Date"], ma100, 'r', label="100-Day MA")
plt.plot(df["Date"], ma200, 'g', label="200-Day MA")
plt.legend()
plt.title(f"{ticker} — Moving Averages")
st.pyplot(plt)


# PREDICTION MODEL

st.subheader("📌 Linear Regression Model Prediction")

x = df[['Open']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test).flatten()

st.write("### 🔹 Sample Predictions")
st.write(pd.DataFrame({
    "Actual": y_test.values.flatten(),
    "Predicted": y_pred
}).head())


# PLOT 3 – ACTUAL VS PREDICTED (LAST 20%)

st.subheader("📌 Actual vs Predicted Close Prices (Last 20%)")

test_indices = df.index[len(X_train):]

plt.figure(figsize=(12, 6))
plt.plot(test_indices, y_test.values, color='green', label='Actual Close Price')
plt.plot(test_indices, y_pred, color='red', linestyle='--', label='Prediction')
plt.xlabel("Index")
plt.ylabel("Close Price")
plt.title(f"Actual vs Predicted — {ticker}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
st.pyplot(plt)


plt.figure(figsize=(12, 6))
plt.plot(test_indices[:100], y_test.values[:100], color='green', label='Actual Close Price')
plt.plot(test_indices[:100], y_pred[:100], color='red', linestyle='--', label='Prediction')
plt.xlabel("Index")
plt.ylabel("Close Price")
plt.title(f"Actual vs Predicted — {ticker} First 100 days")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
st.pyplot(plt)




# 📌 NEXT 10 DAYS PREDICTION (Linear Regression Version)
num_days = st.selectbox(
    "Select No. of days to predict",
    ["1 year", "10 years", "6 months", "10 days"]

)


st.subheader(f"📌 Next {num_days} Days Predicted Close Prices")
if num_days=='1 year':
    num_days=365
elif num_days=='10 years':
    num_days=3650
elif num_days=='6 months':
    num_days==30*6
elif num_days=='10 days':
    num_days=10

# Always start with last OPEN price as a float
current_open = float(df["Open"].iloc[-1])

future_predictions = []


# Generate 10 future dates correctly
future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=num_days
)

# Predict next 10 days
for i in range(num_days):
    next_close = float(model.predict([[current_open]])[0])
    future_predictions.append(next_close)

    # Next day's OPEN = today's predicted CLOSE
    current_open = next_close

# --- SAFETY CHECK ---
# This guarantees same length
assert len(future_dates) == len(future_predictions)

# Create DataFrame safely
future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close": future_predictions
})

st.write(future_df)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(future_df["Date"], future_df["Predicted Close"], marker='o')
plt.title(f"Next {num_days} Days Predicted Close Prices")
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.grid(True, linestyle='--', alpha=0.5)
st.pyplot(plt)


st.success("✔ All plots generated successfully!")