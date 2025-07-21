import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.random.set_seed(42)
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict next-day stock prices using advanced machine learning models")

# Sidebar for user inputs
st.sidebar.header("📊 Stock Selection & Parameters")

# Stock symbol input
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, TSLA)", value="AAPL")
symbol = symbol.upper()

# Date range selection
st.sidebar.subheader("📅 Historical Data Range")
end_date = datetime.now()
start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=365))
days_to_predict = st.sidebar.selectbox("Prediction Horizon", [1, 5, 10], index=0)

# Model selection
st.sidebar.subheader("🧠 ML Model Selection")
model_choice = st.sidebar.selectbox("Choose Prediction Model",
                                    ["Random Forest", "Linear Regression", "XGBoost",
                                     "LSTM Neural Network", "Gradient Boosting", "Advanced Ensemble"])

# Advanced parameters
st.sidebar.subheader("⚙️ Advanced Parameters")
lookback_days = st.sidebar.slider("Lookback Days for Features", 5, 30, 14)
use_technical_indicators = st.sidebar.checkbox("Use Technical Indicators", value=True)


def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    # Moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()

    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

    # Volatility
    data['Volatility'] = data['Close'].rolling(window=10).std()

    return data


def create_features(data, lookback_days=14, use_technical=True):
    """Create features for ML model"""
    features = []

    # Price-based features
    for i in range(1, lookback_days + 1):
        features.append(f'Close_lag_{i}')
        data[f'Close_lag_{i}'] = data['Close'].shift(i)

        features.append(f'Volume_lag_{i}')
        data[f'Volume_lag_{i}'] = data['Volume'].shift(i)

        features.append(f'Return_lag_{i}')
        data[f'Return_lag_{i}'] = data['Close'].pct_change(i)

    # Technical indicators
    if use_technical:
        tech_features = ['MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'MACD_Signal',
                         'BB_Upper', 'BB_Lower', 'Volatility']
        features.extend(tech_features)

    # Target variable (next day's closing price)
    data['Target'] = data['Close'].shift(-1)

    return data, features


def create_lstm_sequences(data, sequence_length=60):
    """Create sequences for LSTM model"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """Build LSTM neural network model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_model(X_train, y_train, model_type="Random Forest"):
    """Train the selected ML model"""
    scaler = StandardScaler()

    if model_type == "LSTM Neural Network":
        # Special handling for LSTM
        X_train_scaled = scaler.fit_transform(X_train)

        # Create sequences for LSTM
        X_lstm, y_lstm = create_lstm_sequences(X_train_scaled[:, 0])  # Use only close price for LSTM

        if len(X_lstm) < 10:
            st.error("Not enough data for LSTM training. Need at least 70 days of data.")
            return None, None

        # Reshape for LSTM
        X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

        # Build and train LSTM model
        model = build_lstm_model((X_lstm.shape[1], 1))

        with st.spinner("Training LSTM model... This may take a few minutes."):
            model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)

        return model, scaler

    # Scale features for other models
    X_train_scaled = scaler.fit_transform(X_train)

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15,
                                      min_samples_split=5, min_samples_leaf=2)
    elif model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=200, random_state=42, max_depth=8,
                                 learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=8,
                                          learning_rate=0.1, subsample=0.8)
    elif model_type == "Advanced Ensemble":
        # Create multiple models for ensemble
        models = {
            'rf': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15),
            'xgb': xgb.XGBRegressor(n_estimators=200, random_state=42, max_depth=8, learning_rate=0.1),
            'gb': GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=8, learning_rate=0.1),
            'svr': SVR(kernel='rbf', C=100, gamma=0.1)
        }

        # Train all models
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)

        return models, scaler

    # Train single model
    model.fit(X_train_scaled, y_train)
    return model, scaler


def make_prediction(model, scaler, X_test, model_type="Random Forest"):
    """Make predictions"""
    if model_type == "LSTM Neural Network":
        # Special handling for LSTM
        X_test_scaled = scaler.transform(X_test)

        # Create sequences for LSTM
        X_lstm, _ = create_lstm_sequences(X_test_scaled[:, 0])

        if len(X_lstm) == 0:
            # If not enough data for sequences, use the last available data
            X_lstm = X_test_scaled[-60:, 0].reshape(1, -1, 1)
        else:
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

        predictions = model.predict(X_lstm, verbose=0)
        return predictions.flatten()

    # Scale features for other models
    X_test_scaled = scaler.transform(X_test)

    if model_type == "Advanced Ensemble":
        # Average predictions from all models
        predictions = []
        for name, m in model.items():
            pred = m.predict(X_test_scaled)
            predictions.append(pred)

        # Weighted average (you can adjust weights based on model performance)
        weights = [0.3, 0.3, 0.25, 0.15]  # RF, XGB, GB, SVR
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred
    else:
        return model.predict(X_test_scaled)


# Main application logic
if st.sidebar.button("🚀 Predict Stock Price", type="primary"):
    # Fetch data
    with st.spinner("Fetching stock data..."):
        data = fetch_stock_data(symbol, start_date, end_date)

    if data is not None and len(data) > 0:
        # Display basic stock info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Price", f"${data['Close'][-1]:.2f}")
        with col2:
            change = data['Close'][-1] - data['Close'][-2]
            st.metric("Daily Change", f"${change:.2f}", f"{change:.2f}")
        with col3:
            st.metric("Volume", f"{data['Volume'][-1]:,.0f}")
        with col4:
            st.metric("52W High", f"${data['High'].max():.2f}")

        # Calculate technical indicators
        if use_technical_indicators:
            data = calculate_technical_indicators(data)

        # Create features
        with st.spinner("Preparing features..."):
            data, feature_names = create_features(data, lookback_days, use_technical_indicators)

        # Prepare training data
        df_clean = data.dropna()

        if len(df_clean) < 50:
            st.error("Not enough data for training. Please select a longer date range.")
        else:
            # Split data
            train_size = int(len(df_clean) * 0.8)
            train_data = df_clean[:train_size]
            test_data = df_clean[train_size:]

            X_train = train_data[feature_names]
            y_train = train_data['Target']
            X_test = test_data[feature_names]
            y_test = test_data['Target']

            # Train model
            with st.spinner("Training AI model..."):
                model, scaler = train_model(X_train, y_train, model_choice)

            # Make predictions on test set
            test_predictions = make_prediction(model, scaler, X_test, model_choice)

            # Calculate metrics
            mae = mean_absolute_error(y_test, test_predictions)
            mse = mean_squared_error(y_test, test_predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, test_predictions)

            # Display model performance
            st.subheader("📈 Model Performance")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean Absolute Error", f"${mae:.2f}")
            with col2:
                st.metric("RMSE", f"${rmse:.2f}")
            with col3:
                st.metric("R² Score", f"{r2:.3f}")
            with col4:
                accuracy = max(0, (1 - mae / data['Close'].mean()) * 100)
                st.metric("Accuracy", f"{accuracy:.1f}%")

            # Make future prediction
            latest_features = df_clean[feature_names].iloc[-1:].values
            future_price = make_prediction(model, scaler, latest_features, model_choice)[0]

            current_price = data['Close'][-1]
            price_change = future_price - current_price
            price_change_pct = (price_change / current_price) * 100

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Next Day Prediction for {symbol}</h2>
                <h1>${future_price:.2f}</h1>
                <p>Expected change: ${price_change:.2f} ({price_change_pct:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            # Create visualization
            st.subheader("📊 Price Prediction Visualization")

            # Historical prices with prediction
            fig = go.Figure()

            # Historical prices
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue')
            ))

            # Test predictions
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_predictions,
                mode='lines',
                name='Test Predictions',
                line=dict(color='orange', dash='dash')
            ))

            # Future prediction
            future_date = data.index[-1] + timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[data.index[-1], future_date],
                y=[current_price, future_price],
                mode='lines+markers',
                name='Future Prediction',
                line=dict(color='red', width=3),
                marker=dict(size=10)
            ))

            fig.update_layout(
                title=f'{symbol} Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                showlegend=True,
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # Feature importance (for Random Forest)
            if model_choice == "Random Forest":
                st.subheader("🔍 Feature Importance")

                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(10)

                    fig_importance = px.bar(
                        feature_importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Most Important Features'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

            # Risk assessment
            st.subheader("⚠️ Risk Assessment")

            volatility = data['Close'].pct_change().std() * np.sqrt(252)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Volatility Analysis:**")
                st.write(f"- Annual Volatility: {volatility:.2%}")
                st.write(f"- Model Accuracy: {accuracy:.1f}%")

                if volatility > 0.3:
                    st.warning("High volatility stock - predictions may be less reliable")
                elif volatility < 0.15:
                    st.success("Low volatility stock - predictions may be more reliable")
                else:
                    st.info("Medium volatility stock - moderate prediction reliability")

            with col2:
                st.write("**Prediction Confidence:**")
                confidence = min(100, accuracy)
                st.progress(confidence / 100)
                st.write(f"Confidence Level: {confidence:.1f}%")

                if confidence > 80:
                    st.success("High confidence prediction")
                elif confidence > 60:
                    st.warning("Medium confidence prediction")
                else:
                    st.error("Low confidence prediction - use with caution")

            # Disclaimer
            st.markdown("---")
            st.markdown("""
            **⚠️ Important Disclaimer:**
            This prediction is based on historical data and machine learning models. 
            Stock prices are influenced by many factors including market sentiment, news, 
            economic conditions, and other unpredictable events. This tool should not be 
            used as the sole basis for investment decisions. Always consult with financial 
            advisors and do your own research before making investment decisions.
            """)

else:
    # Display sample data and instructions
    st.info("👆 Enter a stock symbol and click 'Predict Stock Price' to get started!")

    # Show popular stocks
    st.subheader("📈 Popular Stock Symbols")
    popular_stocks = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "TSLA": "Tesla Inc.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "NFLX": "Netflix Inc."
    }

    cols = st.columns(4)
    for i, (symbol, name) in enumerate(popular_stocks.items()):
        with cols[i % 4]:
            st.write(f"**{symbol}** - {name}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, scikit-learn, and yfinance")