"""
Dashboard Monitoring Sistem Trading Sinyal
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Konfigurasi
st.set_page_config(
    page_title="Forex & IDX Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MongoDB Connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URL)
db = client[os.getenv("DATABASE_NAME", "forex_idx_signals")]

# API Base URL (FastAPI)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Title
st.title("📊 Forex & IDX Trading Signal Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stock.png", width=80)
    st.title("Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard Overview", "📈 Active Signals", "📊 Backtest Results", 
         "🤖 Model Management", "⚙️ Optimization", "📝 Trade History", "📰 News Sentiment"]
    )
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**System Status:** 🟢 Online")

# Helper Functions
def fetch_signals():
    """Fetch active signals from MongoDB"""
    signals = list(db.signals.find({}, {"_id": 0}).sort("probability", -1).limit(20))
    return signals

def fetch_backtest_results(symbol=None):
    """Fetch backtest results"""
    query = {}
    if symbol:
        query["symbol"] = symbol
    results = list(db.backtest_results.find(query, {"_id": 0}).sort("timestamp", -1).limit(10))
    return results

def fetch_model_metadata():
    """Fetch model metadata"""
    models = list(db.model_metadata.find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
    active_config = db.config.find_one({"key": "active_model"})
    active_model = None
    if active_config:
        active_model = db.model_metadata.find_one({"model_path": active_config["value"]})
    return models, active_model

def fetch_optimization_results():
    """Fetch latest optimization results"""
    results = list(db.optimization_results.find({}, {"_id": 0}).sort("timestamp", -1).limit(5))
    return results

def fetch_trades(symbol=None, limit=50):
    """Fetch trade history"""
    query = {}
    if symbol:
        query["symbol"] = symbol
    trades = list(db.trades_collection.find(query, {"_id": 0}).sort("exit_date", -1).limit(limit))
    return trades

def plot_equity_curve(equity_data):
    """Plot equity curve and drawdown"""
    if not equity_data:
        return None
    df = pd.DataFrame(equity_data, columns=['date', 'equity'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=('Equity Curve', 'Drawdown'))
    
    # Equity curve
    fig.add_trace(go.Scatter(x=df.index, y=df['equity'], 
                             mode='lines', name='Equity',
                             line=dict(color='green', width=2)), row=1, col=1)
    
    # Drawdown
    cumulative = df['equity']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    fig.add_trace(go.Scatter(x=df.index, y=drawdown, 
                             mode='lines', name='Drawdown',
                             fill='tozeroy', line=dict(color='red', width=1)), row=2, col=1)
    
    fig.update_layout(height=600, title_text="Equity Curve & Drawdown Analysis")
    fig.update_yaxes(title_text="Equity (Rp)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig

def plot_performance_metrics(metrics):
    """Plot performance metrics as gauge charts"""
    if not metrics:
        return None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_win = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics.get('win_rate', 0) * 100,
            title={'text': "Win Rate"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "green"},
                   'steps': [
                       {'range': [0, 40], 'color': "red"},
                       {'range': [40, 60], 'color': "orange"},
                       {'range': [60, 100], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75,
                                 'value': 50}}))
        st.plotly_chart(fig_win, use_container_width=True)
    
    with col2:
        fig_sharpe = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics.get('sharpe_ratio', 0),
            title={'text': "Sharpe Ratio"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 3]},
                   'bar': {'color': "blue"},
                   'steps': [
                       {'range': [0, 1], 'color': "red"},
                       {'range': [1, 2], 'color': "orange"},
                       {'range': [2, 3], 'color': "green"}]}))
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    with col3:
        fig_profit = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics.get('profit_factor', 0),
            title={'text': "Profit Factor"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 3]},
                   'bar': {'color': "purple"},
                   'steps': [
                       {'range': [0, 1], 'color': "red"},
                       {'range': [1, 1.5], 'color': "orange"},
                       {'range': [1.5, 3], 'color': "green"}]}))
        st.plotly_chart(fig_profit, use_container_width=True)

def plot_signal_distribution(signals):
    """Plot distribution of signals by action and bias"""
    if not signals:
        return None
    
    df = pd.DataFrame(signals)
    action_counts = df['action'].value_counts()
    bias_counts = df['bias'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_action = px.pie(values=action_counts.values, names=action_counts.index, 
                            title="Signal Actions", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_action, use_container_width=True)
    
    with col2:
        fig_bias = px.pie(values=bias_counts.values, names=bias_counts.index, 
                          title="Market Bias", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_bias, use_container_width=True)

def plot_top_signals(signals):
    """Plot top signals by probability"""
    if not signals:
        return None
    
    df = pd.DataFrame(signals)
    df_top = df.nlargest(10, 'probability')
    
    colors = ['green' if x == 'buy' else 'red' if x == 'sell' else 'gray' for x in df_top['action']]
    
    fig = go.Figure(go.Bar(
        x=df_top['symbol'],
        y=df_top['probability'],
        text=df_top['probability'],
        textposition='auto',
        marker_color=colors,
        name='Probability'
    ))
    
    fig.update_layout(title="Top 10 Signals by Probability",
                      xaxis_title="Symbol",
                      yaxis_title="Probability (%)",
                      height=400)
    return fig

# Page Content
if page == "🏠 Dashboard Overview":
    st.header("📊 Dashboard Overview")
    
    # Get data
    signals = fetch_signals()
    backtest_results = fetch_backtest_results()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    active_signals = len([s for s in signals if s['action'] != 'hold'])
    buy_signals = len([s for s in signals if s['action'] == 'buy'])
    sell_signals = len([s for s in signals if s['action'] == 'sell'])
    
    with col1:
        st.metric("Active Signals", active_signals, delta=None)
    with col2:
        st.metric("Buy Signals", buy_signals, delta=None)
    with col3:
        st.metric("Sell Signals", sell_signals, delta=None)
    with col4:
        st.metric("Instruments Tracked", len(signals))
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Signal Distribution")
        plot_signal_distribution(signals)
    
    with col2:
        st.subheader("Top Signals by Probability")
        fig = plot_top_signals(signals)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent Backtest Performance
    st.subheader("Recent Backtest Performance")
    if backtest_results:
        latest = backtest_results[0]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{latest.get('total_return', 0):.2%}")
        with col2:
            st.metric("Win Rate", f"{latest.get('win_rate', 0):.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{latest.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{latest.get('max_drawdown', 0):.2%}")
        
        # Equity curve if available
        if 'daily_equity' in latest:
            fig = plot_equity_curve(latest['daily_equity'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No backtest results available. Run a backtest to see performance metrics.")

elif page == "📈 Active Signals":
    st.header("📈 Active Trading Signals")
    
    signals = fetch_signals()
    
    if signals:
        df_signals = pd.DataFrame(signals)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            action_filter = st.multiselect("Filter by Action", options=['buy', 'sell', 'hold'], default=['buy', 'sell'])
        with col2:
            bias_filter = st.multiselect("Filter by Bias", options=['bullish', 'bearish', 'neutral'], default=['bullish', 'bearish'])
        
        df_filtered = df_signals[df_signals['action'].isin(action_filter) & df_signals['bias'].isin(bias_filter)]
        
        # Display signals
        for _, signal in df_filtered.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 3])
                with col1:
                    st.subheader(f"**{signal['symbol']}**")
                with col2:
                    action_color = "green" if signal['action'] == 'buy' else "red" if signal['action'] == 'sell' else "gray"
                    st.markdown(f"<h3 style='color:{action_color}'>{signal['action'].upper()}</h3>", unsafe_allow_html=True)
                with col3:
                    st.metric("Prob", f"{signal['probability']}%")
                with col4:
                    st.metric("RR", signal['risk_reward'])
                with col5:
                    st.text(f"Entry: {signal['entry_zone']:,.0f}")
                    st.text(f"SL: {signal['stop_loss_1']:,.0f} | TP: {signal['take_profit_1']:,.0f}")
                
                with st.expander("Details"):
                    st.write(f"**Bias:** {signal['bias']}")
                    st.write(f"**Action Type:** {signal['action_type']}")
                    st.write(f"**Stop Loss 2:** {signal['stop_loss_2']:,.0f}")
                    st.write(f"**Take Profit 2:** {signal['take_profit_2']:,.0f}")
                    st.write(f"**Notes:** {signal['notes']}")
                st.markdown("---")
    else:
        st.info("No active signals found.")

elif page == "📊 Backtest Results":
    st.header("📊 Backtest Results")
    
    # Get unique symbols
    symbols = list(db.backtest_results.distinct("symbol"))
    selected_symbol = st.selectbox("Select Symbol", ["All"] + symbols)
    
    results = fetch_backtest_results(selected_symbol if selected_symbol != "All" else None)
    
    if results:
        for result in results:
            st.subheader(f"{result['symbol']} - {result['start_date']} to {result['end_date']}")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{result.get('total_return', 0):.2%}")
            with col2:
                st.metric("CAGR", f"{result.get('cagr', 0):.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}")
            with col4:
                st.metric("Max Drawdown", f"{result.get('max_drawdown', 0):.2%}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Win Rate", f"{result.get('win_rate', 0):.2%}")
            with col2:
                st.metric("Profit Factor", f"{result.get('profit_factor', 0):.2f}")
            with col3:
                st.metric("Num Trades", result.get('num_trades', 0))
            with col4:
                st.metric("Expectancy", f"Rp {result.get('expectancy', 0):,.0f}")
            
            # Additional metrics
            with st.expander("Advanced Metrics"):
                st.write(f"**Avg Win:** Rp {result.get('avg_win', 0):,.0f} ({result.get('avg_win_percent', 0):.2%})")
                st.write(f"**Avg Loss:** Rp {result.get('avg_loss', 0):,.0f} ({result.get('avg_loss_percent', 0):.2%})")
                st.write(f"**Win/Loss Ratio:** {result.get('win_loss_ratio', 0):.2f}")
                st.write(f"**Max Consecutive Wins:** {result.get('max_consecutive_wins', 0)}")
                st.write(f"**Max Consecutive Losses:** {result.get('max_consecutive_losses', 0)}")
                st.write(f"**Recovery Factor:** {result.get('recovery_factor', 0):.2f}")
                st.write(f"**Avg Drawdown:** {result.get('avg_drawdown', 0):.2%}")
                st.write(f"**Max Drawdown Duration:** {result.get('max_drawdown_duration_days', 0)} days")
                st.write(f"**Avg Drawdown Duration:** {result.get('avg_drawdown_duration_days', 0):.1f} days")
            
            # Equity curve
            if 'daily_equity' in result:
                fig = plot_equity_curve(result['daily_equity'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    else:
        st.info("No backtest results found. Run a backtest to see results.")

elif page == "🤖 Model Management":
    st.header("🤖 Machine Learning Model Management")
    
    models, active_model = fetch_model_metadata()
    
    # Active model info
    st.subheader("Active Model")
    if active_model:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Path", active_model['model_path'].split('/')[-1])
        with col2:
            st.metric("Accuracy", f"{active_model['accuracy']:.2%}")
        with col3:
            st.metric("Trained", active_model['timestamp'].strftime('%Y-%m-%d'))
        
        with st.expander("Model Details"):
            st.write(f"**Input Dimension:** {active_model.get('input_dim', 'N/A')}")
            st.write(f"**Feature Columns:** {active_model.get('feature_cols', [])}")
            st.write(f"**Target Days:** {active_model.get('target_days', 5)}")
            st.write(f"**Training Period:** {active_model.get('start_date', 'N/A')} to {active_model.get('end_date', 'N/A')}")
    else:
        st.info("No active model found. Train a model to activate it.")
    
    st.markdown("---")
    
    # Model list
    st.subheader("Available Models")
    if models:
        df_models = pd.DataFrame(models)
        df_models['timestamp'] = pd.to_datetime(df_models['timestamp'])
        df_models = df_models.sort_values('timestamp', ascending=False)
        
        for _, model in df_models.iterrows():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(f"**{model['model_path'].split('/')[-1]}**")
            with col2:
                st.write(f"Acc: {model['accuracy']:.2%}")
            with col3:
                st.write(model['timestamp'].strftime('%Y-%m-%d'))
            with col4:
                if active_model and model['model_path'] == active_model['model_path']:
                    st.markdown("✅ **Active**")
                else:
                    if st.button("Activate", key=model['model_path']):
                        # API call to activate model
                        response = requests.post(f"{API_URL}/model/activate", params={"model_path": model['model_path']})
                        if response.status_code == 200:
                            st.success(f"Model {model['model_path']} activated!")
                            st.rerun()
                        else:
                            st.error("Failed to activate model")
            st.markdown("---")
    else:
        st.info("No models available. Train a model to see it here.")
    
    # Train new model
    st.subheader("Train New Model")
    with st.form("train_model"):
        col1, col2 = st.columns(2)
        with col1:
            symbols_input = st.text_input("Symbols (comma separated)", "BBCA.JK,BBRI.JK,ASII.JK")
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
            epochs = st.number_input("Epochs", min_value=10, max_value=500, value=100)
        
        if st.form_submit_button("Start Training"):
            symbols_list = [s.strip() for s in symbols_input.split(",")]
            with st.spinner("Training model... This may take a few minutes."):
                response = requests.post(
                    f"{API_URL}/model/train",
                    json={
                        "symbols": symbols_list,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "epochs": epochs
                    }
                )
                if response.status_code == 200:
                    st.success("Model training completed!")
                    st.rerun()
                else:
                    st.error(f"Training failed: {response.text}")

elif page == "⚙️ Optimization":
    st.header("⚙️ Parameter Optimization")
    
    optimization_results = fetch_optimization_results()
    
    # Start new optimization
    st.subheader("Start New Optimization")
    with st.form("optimization_form"):
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol", "BBCA.JK")
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=180))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
            metric = st.selectbox("Optimization Metric", ["sharpe_ratio", "total_return", "profit_factor", "win_rate"])
        
        st.write("Parameter Grid:")
        param_grid = {
            "lookback_ob": [20, 30],
            "volume_mult_ob": [1.5, 2.0],
            "risk_per_trade": [0.01, 0.02],
            "sl_mult": [0.98, 0.97],
            "tp_mult": [1.04, 1.05]
        }
        st.json(param_grid)
        
        if st.form_submit_button("Start Optimization"):
            with st.spinner("Running optimization... This may take several minutes."):
                response = requests.post(
                    f"{API_URL}/optimize/start",
                    json={
                        "symbol": symbol,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "param_grid": param_grid,
                        "metric": metric
                    }
                )
                if response.status_code == 200:
                    st.success(f"Optimization started! ID: {response.json()['optimization_id']}")
                else:
                    st.error(f"Optimization failed: {response.text}")
    
    st.markdown("---")
    
    # Previous optimization results
    st.subheader("Previous Optimization Results")
    if optimization_results:
        for opt in optimization_results:
            with st.expander(f"Optimization {opt.get('timestamp', 'N/A')} - {opt.get('symbol', 'N/A')}"):
                if opt.get('best_params'):
                    st.write("**Best Parameters:**")
                    st.json(opt['best_params'])
                if opt.get('best_metrics'):
                    st.write("**Best Metrics:**")
                    cols = st.columns(4)
                    metrics = opt['best_metrics']
                    with cols[0]:
                        st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
                    with cols[1]:
                        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    with cols[2]:
                        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
                    with cols[3]:
                        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    else:
        st.info("No optimization results found.")

elif page == "📝 Trade History":
    st.header("📝 Trade History")
    
    # Get unique symbols
    symbols = list(db.trades_collection.distinct("symbol"))
    selected_symbol = st.selectbox("Select Symbol", ["All"] + symbols)
    
    trades = fetch_trades(selected_symbol if selected_symbol != "All" else None)
    
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        df_trades['profit_pct'] = df_trades['profit_pct'] * 100
        
        # Summary stats
        total_profit = df_trades['profit'].sum()
        win_rate = (df_trades['profit'] > 0).mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(df_trades))
        with col2:
            st.metric("Total Profit/Loss", f"Rp {total_profit:,.0f}")
        with col3:
            st.metric("Win Rate", f"{win_rate:.2%}")
        with col4:
            st.metric("Avg Profit/Trade", f"Rp {df_trades['profit'].mean():,.0f}")
        
        # Trades table
        st.subheader("Trade List")
        display_df = df_trades[['symbol', 'action', 'entry_date', 'exit_date', 
                                 'entry_price', 'exit_price', 'profit', 'profit_pct', 
                                 'exit_reason']].copy()
        display_df['profit_pct'] = display_df['profit_pct'].apply(lambda x: f"{x:.2f}%")
        display_df['profit'] = display_df['profit'].apply(lambda x: f"Rp {x:,.0f}")
        display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"{x:,.0f}")
        display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Profit distribution
        st.subheader("Profit Distribution")
        fig = px.histogram(df_trades, x='profit', nbins=30, 
                          title="Trade Profit Distribution",
                          labels={'profit': 'Profit (Rp)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative profit
        st.subheader("Cumulative Profit")
        df_trades['cumulative_profit'] = df_trades['profit'].cumsum()
        fig = go.Figure(go.Scatter(x=df_trades['exit_date'], y=df_trades['cumulative_profit'],
                                   mode='lines', name='Cumulative Profit',
                                   line=dict(color='green', width=2)))
        fig.update_layout(title="Cumulative Profit Over Time",
                          xaxis_title="Exit Date",
                          yaxis_title="Cumulative Profit (Rp)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades found.")

elif page == "📰 News Sentiment":
    st.header("📰 News Sentiment Analysis")
    
    symbol = st.text_input("Symbol", "BBCA.JK")
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing news sentiment..."):
            response = requests.get(f"{API_URL}/sentiment/{symbol}")
            if response.status_code == 200:
                data = response.json()
                
                st.subheader(f"Sentiment Analysis for {symbol}")
                
                # Overall sentiment
                avg_score = data.get('avg_score', 0)
                sentiment_color = "green" if avg_score > 0 else "red" if avg_score < 0 else "gray"
                st.markdown(f"<h2 style='color:{sentiment_color}'>Overall Sentiment: {avg_score:.2f}</h2>", unsafe_allow_html=True)
                
                # News items
                st.subheader("Analyzed News")
                for news in data.get('sentiments', []):
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{news.get('title', 'N/A')[:100]}...**")
                        with col2:
                            sentiment = news.get('groq_sentiment', news.get('model_sentiment', 'neutral'))
                            sentiment_color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "gray"
                            st.markdown(f"<span style='color:{sentiment_color}'>{sentiment.upper()}</span>", unsafe_allow_html=True)
                        if news.get('key_points'):
                            st.write(f"*Key points:* {', '.join(news['key_points'])}")
                        st.markdown("---")
            else:
                st.error(f"Failed to analyze sentiment: {response.text}")

# Auto-refresh
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()