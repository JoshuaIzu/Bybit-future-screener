import streamlit as st
import numpy as np
import pandas as pd
import ccxt
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from tabulate import tabulate
from datetime import datetime, timedelta
import sqlite3

# Set page config
st.set_page_config(
    page_title="Bybit Futures Screener",
    page_icon="📊",
    layout="wide",
)

# Title and description
st.title("📈 Bybit Futures Market Screener")
st.markdown("Track and analyze cryptocurrency futures markets on Bybit")

# Initialize Bybit connection
@st.cache_resource()
def get_exchange():
    return ccxt.bybit({
        'options': {
            'defaultType': 'future',
            'recvWindow': 10000,
        },
        'urls': {
            'api': {
                'public': 'https://api.bybit.com',  # Corrected endpoint
                'private': 'https://api.bybit.com'
            }
        },
    
    })
exchange = get_exchange()

# Configuration
BASE_VOL = 0.35
VOL_MULTIPLIER = 1.5
MIN_LIQUIDITY = 5000000

# Database functions for state management
def load_state_from_db():
    conn = sqlite3.connect('trading_state.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS app_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    
    # Get active trades
    cursor.execute('SELECT value FROM app_state WHERE key = "active_trades"')
    active_trades_row = cursor.fetchone()
    active_trades = json.loads(active_trades_row[0]) if active_trades_row else {}
    
    # Get completed trades
    cursor.execute('SELECT value FROM app_state WHERE key = "completed_trades"')
    completed_trades_row = cursor.fetchone()
    completed_trades = json.loads(completed_trades_row[0]) if completed_trades_row else []
    
    conn.close()
    return active_trades, completed_trades

def save_state_to_db(active_trades, completed_trades):
    conn = sqlite3.connect('trading_state.db')
    cursor = conn.cursor()
    
    # Save active trades
    cursor.execute('''
    INSERT OR REPLACE INTO app_state (key, value)
    VALUES (?, ?)
    ''', ('active_trades', json.dumps(active_trades)))
    
    # Save completed trades
    cursor.execute('''
    INSERT OR REPLACE INTO app_state (key, value)
    VALUES (?, ?)
    ''', ('completed_trades', json.dumps(completed_trades)))
    
    conn.commit()
    conn.close()

# Sidebar for parameters
with st.sidebar:
    st.header("Parameters")
    BASE_VOL = st.slider("Base Volume Threshold", 0.1, 2.0, 0.35, 0.05)
    VOL_MULTIPLIER = st.slider("Volume Multiplier", 1.0, 3.0, 1.5, 0.1)
    MIN_LIQUIDITY = st.number_input("Minimum Liquidity (USD)", 1000000, 20000000, 5000000, 1000000)

class ForwardTester:
    def __init__(self):
        self.active_trades = {}
        self.completed_trades = []
        self.load_state()
        
    def load_state(self):
        """Load previous trading state from database"""
        try:
            self.active_trades, self.completed_trades = load_state_from_db()
        except Exception as e:
            st.warning(f"Could not load previous state: {str(e)}. Starting fresh.")
    
    def save_state(self):
        """Save current trading state to database"""
        try:
            save_state_to_db(self.active_trades, self.completed_trades)
        except Exception as e:
            st.error(f"Error saving state to database: {str(e)}")
    
    def execute_trades(self, signals):
        """Execute new trades based on signals"""
        executed = []
        for signal in signals:
            symbol = signal['Symbol']
            
            # Skip if already in active trades
            if symbol in self.active_trades:
                continue
                
            if signal['Signal'] != "HOLD":
                self.active_trades[symbol] = {
                    'Symbol': symbol,
                    'entry_price': signal['Price'],
                    'entry_time': datetime.now().isoformat(),
                    'direction': signal['Signal'],
                    'tp_price': float(signal['TP']) if signal['TP'] != "-" else None,
                    'sl_price': float(signal['SL']) if signal['SL'] != "-" else None,
                    'reason': signal['Reason'],
                    'status': 'OPEN'
                }
                executed.append(f"📝 New {signal['Signal']} trade for {symbol} at {signal['Price']}")
        
        self.save_state()
        return executed
    
    def update_trades(self):
        """Check open trades for TP/SL hits"""
        to_remove = []
        updates = []
        
        for symbol, trade in self.active_trades.items():
            try:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                entry_price = trade['entry_price']
                
                # Check for TP/SL
                if trade['direction'] == "LONG":
                    if trade['tp_price'] and current_price >= trade['tp_price']:
                        trade['exit_reason'] = "TP Hit"
                    elif trade['sl_price'] and current_price <= trade['sl_price']:
                        trade['exit_reason'] = "SL Hit"
                elif trade['direction'] == "SHORT":
                    if trade['tp_price'] and current_price <= trade['tp_price']:
                        trade['exit_reason'] = "TP Hit"
                    elif trade['sl_price'] and current_price >= trade['sl_price']:
                        trade['exit_reason'] = "SL Hit"
                
                # Mark for removal if closed
                if 'exit_reason' in trade:
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['status'] = 'CLOSED'
                    trade['pct_change'] = ((current_price - entry_price)/entry_price)*100 if trade['direction'] == "LONG" else ((entry_price - current_price)/entry_price)*100
                    self.completed_trades.append(trade)
                    to_remove.append(symbol)
                    updates.append(f"✅ Trade closed: {symbol} | Reason: {trade['exit_reason']} | PnL: {trade['pct_change']:.2f}%")
            
            except Exception as e:
                updates.append(f"Error updating {symbol}: {str(e)}")
        
        # Remove closed trades
        for symbol in to_remove:
            self.active_trades.pop(symbol)
        
        self.save_state()
        return updates
    
    def get_performance_report(self):
        """Generate performance report"""
        if not self.completed_trades:
            return "No completed trades yet", pd.DataFrame()
            
        df = pd.DataFrame(self.completed_trades)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds()/3600
        else:
            df['duration'] = 0
        
        stats = {
            'total_trades': len(df),
            'win_rate': len(df[df['pct_change'] > 0])/len(df) if len(df) > 0 else 0,
            'avg_pnl': df['pct_change'].mean() if 'pct_change' in df.columns else 0,
            'avg_duration_hours': df['duration'].mean() if 'duration' in df.columns else 0
        }
        
        return stats, df

    def reset_all_trades(self):
        """Reset all trades in the database"""
        self.active_trades = {}
        self.completed_trades = []
        self.save_state()
        return "All trades have been reset"

# Function to fetch all markets and filter
@st.cache_data(ttl=600)
def fetch_all_markets():
    """Fetch all USDT perpetual markets from Bybit"""
    try:
        markets = exchange.load_markets()
        usdt_perps = [
            symbol for symbol, market in markets.items()
            if market['quote'] == 'USDT' and market['type'] == 'swap'
        ]
        
        results = []
        for symbol in usdt_perps:
            try:
                ticker = exchange.fetch_ticker(symbol)
                volume_24h = ticker['quoteVolume']
                
                # Skip low liquidity markets
                if volume_24h < MIN_LIQUIDITY:
                    continue
                
                results.append({
                    'symbol': symbol,
                    'last_price': ticker['last'],
                    'volume_24h': volume_24h,
                    'change_24h': ticker['percentage']
                })
            except Exception as e:
                pass
                
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error fetching markets: {str(e)}")
        return pd.DataFrame()

# Function to fetch OHLCV data
@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, timeframe='1h', limit=500):
    """Fetch OHLCV data for a symbol"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
        return None

# Create a separate database connection for caching market data
def init_market_cache():
    conn = sqlite3.connect('market_cache.db')
    cursor = conn.cursor()
    
    # Create markets table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_cache (
        symbol TEXT PRIMARY KEY,
        data TEXT,
        timestamp INTEGER
    )
    ''')
    
    # Create OHLCV table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ohlcv_cache (
        symbol TEXT,
        timeframe TEXT,
        data TEXT,
        timestamp INTEGER,
        PRIMARY KEY (symbol, timeframe)
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize cache tables
init_market_cache()

# Function to analyze and generate signals
def generate_signals(markets_df=None):
    """Generate trading signals based on volume analysis"""
    if markets_df is None:
        markets_df = fetch_all_markets()
    
    signals = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (index, market) in enumerate(markets_df.iterrows()):
        symbol = market['symbol']
        status_text.text(f"Analyzing {symbol}...")
        
        # Fetch hourly data
        df = fetch_ohlcv(symbol, timeframe='1h', limit=168)  # 7 days
        if df is None or len(df) < 24:
            continue
            
        # Calculate volume metrics
        avg_vol = df['volume'].mean()
        recent_vol = df['volume'].iloc[-1]
        vol_surge = recent_vol / avg_vol if avg_vol > 0 else 0
        
        signal = "HOLD"
        reason = ""
        tp = "-"
        sl = "-"
        
        # Simple volume surge strategy
        if vol_surge >= VOL_MULTIPLIER and recent_vol > BASE_VOL * avg_vol:
            # Check price action for direction
            recent_change = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
            
            if recent_change > 0.01:  # 1% up
                signal = "LONG"
                reason = f"Vol surge {vol_surge:.2f}x with bullish price action"
                tp = str(round(df['close'].iloc[-1] * 1.05, 4))  # 5% TP
                sl = str(round(df['close'].iloc[-1] * 0.97, 4))  # 3% SL
            elif recent_change < -0.01:  # 1% down
                signal = "SHORT"
                reason = f"Vol surge {vol_surge:.2f}x with bearish price action"
                tp = str(round(df['close'].iloc[-1] * 0.95, 4))  # 5% TP
                sl = str(round(df['close'].iloc[-1] * 1.03, 4))  # 3% SL
        
        signals.append({
            'Symbol': symbol,
            'Price': df['close'].iloc[-1],
            'Signal': signal,
            'Volume 24h': market['volume_24h'],
            'Vol Surge': vol_surge,
            'Change 24h': market['change_24h'],
            'Reason': reason,
            'TP': tp,
            'SL': sl
        })
        
        # Update progress
        progress_bar.progress((i + 1) / len(markets_df))
    
    progress_bar.empty()
    status_text.empty()
    
    return signals

# Initialize the forward tester
tester = ForwardTester()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market Scanner", "Active Trades", "Completed Trades", "Performance", "Database"])

# Tab 1: Market Scanner
with tab1:
    if st.button("Scan Markets"):
        with st.spinner("Scanning all Bybit perpetual markets..."):
            markets_df = fetch_all_markets()
            st.session_state.signals = generate_signals(markets_df)
    
    if 'signals' in st.session_state and st.session_state.signals:
        signals_df = pd.DataFrame(st.session_state.signals)
        
        # Filter options
        signal_filter = st.multiselect("Filter by Signal", 
                                       options=['LONG', 'SHORT', 'HOLD'], 
                                       default=['LONG', 'SHORT'])
        
        filtered_df = signals_df[signals_df['Signal'].isin(signal_filter)]
        
        # Display signals table
        st.dataframe(filtered_df, use_container_width=True)
        
        # Execute trades button
        if st.button("Execute Selected Signals"):
            actionable_signals = [s for s in st.session_state.signals if s['Signal'] != "HOLD" and s['Signal'] in signal_filter]
            results = tester.execute_trades(actionable_signals)
            for result in results:
                st.success(result)

# Tab 2: Active Trades
with tab2:
    st.header("Active Trades")
    
    if st.button("Update Trades"):
        updates = tester.update_trades()
        for update in updates:
            st.info(update)
    
    if tester.active_trades:
        # Convert active trades to DataFrame for display
        active_df = pd.DataFrame.from_dict(tester.active_trades, orient='index')
        
        # Add current price and P&L columns
        for idx, row in active_df.iterrows():
            try:
                ticker = exchange.fetch_ticker(idx)
                current_price = ticker['last']
                entry_price = row['entry_price']
                
                active_df.at[idx, 'current_price'] = current_price
                
                # Calculate unrealized P&L
                if row['direction'] == "LONG":
                    pnl = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pnl = ((entry_price - current_price) / entry_price) * 100
                    
                active_df.at[idx, 'unrealized_pnl'] = pnl
            except:
                active_df.at[idx, 'current_price'] = "Error"
                active_df.at[idx, 'unrealized_pnl'] = 0
        
        # Display active trades
        st.dataframe(active_df, use_container_width=True)
        
        # Individual trade charts
        st.subheader("Individual Trade Charts")
        selected_trade = st.selectbox("Select a trade to view", list(tester.active_trades.keys()))
        
        if selected_trade:
            timeframe = st.selectbox("Timeframe", ['1h', '4h', '1d'], index=0)
            ohlcv_df = fetch_ohlcv(selected_trade, timeframe=timeframe, limit=100)
            
            if ohlcv_df is not None:
                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=ohlcv_df['timestamp'],
                    open=ohlcv_df['open'],
                    high=ohlcv_df['high'],
                    low=ohlcv_df['low'],
                    close=ohlcv_df['close'],
                    name="OHLC"
                )])
                
                # Add entry, TP and SL lines
                trade_data = tester.active_trades[selected_trade]
                
                fig.add_hline(y=trade_data['entry_price'], line_width=1, line_dash="dash", 
                              line_color="yellow", annotation_text="Entry")
                
                if trade_data['tp_price']:
                    fig.add_hline(y=trade_data['tp_price'], line_width=1, line_dash="dash", 
                                 line_color="green", annotation_text="TP")
                    
                if trade_data['sl_price']:
                    fig.add_hline(y=trade_data['sl_price'], line_width=1, line_dash="dash", 
                                 line_color="red", annotation_text="SL")
                
                # Layout
                fig.update_layout(
                    title=f"{selected_trade} - {timeframe} Chart",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active trades at the moment.")

# Tab 3: Completed Trades
with tab3:
    st.header("Completed Trades")
    
    # Reset button with confirmation
    if st.button("Reset All Trades"):
        if st.checkbox("I confirm I want to reset all trades"):
            result = tester.reset_all_trades()
            st.success(result)
    
    # Get completed trades
    stats, completed_df = tester.get_performance_report()
    
    if len(completed_df) > 0:
        # Display completed trades
        st.dataframe(completed_df, use_container_width=True)
        
        # Trade outcomes pie chart
        if 'pct_change' in completed_df.columns:
            fig = px.pie(
                names=['Winning Trades', 'Losing Trades'],
                values=[
                    len(completed_df[completed_df['pct_change'] > 0]),
                    len(completed_df[completed_df['pct_change'] <= 0])
                ],
                color=['green', 'red'],
                title="Trade Outcomes"
            )
            st.plotly_chart(fig)
    else:
        st.info("No completed trades yet.")

# Tab 4: Performance
with tab4:
    st.header("Trading Performance")
    
    stats, completed_df = tester.get_performance_report()
    
    if isinstance(stats, dict):
        stats_df = pd.DataFrame([stats])
        st.dataframe(stats_df, use_container_width=True)
        
        # Show performance charts if we have data
        if len(completed_df) > 0 and 'pct_change' in completed_df.columns:
            # Cumulative performance chart
            if 'exit_time' in completed_df.columns:
                completed_df_sorted = completed_df.sort_values('exit_time')
                completed_df_sorted['cumulative_pnl'] = completed_df_sorted['pct_change'].cumsum()
                
                fig = px.line(
                    completed_df_sorted,
                    x='exit_time', 
                    y='cumulative_pnl',
                    title="Cumulative P&L (%)",
                    labels={'exit_time': 'Date', 'cumulative_pnl': 'Cumulative P&L (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # P&L distribution
            fig = px.histogram(
                completed_df,
                x='pct_change',
                nbins=20,
                title="P&L Distribution",
                labels={'pct_change': 'P&L (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(stats)  # This will display "No completed trades yet"

# Tab 5: Database Management
with tab5:
    st.header("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Database Information")
        if os.path.exists('trading_state.db'):
            conn = sqlite3.connect('trading_state.db')
            cursor = conn.cursor()
            
            # Get database size
            cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            conn.close()
            
            st.write(f"Database Size: {db_size/1024:.2f} KB")
            st.write("Tables:")
            for table in tables:
                st.write(f"- {table[0]}")
        else:
            st.write("Database file not found.")
    
    with col2:
        st.subheader("Database Actions")
        if st.button("Backup Database"):
            try:
                import shutil
                backup_file = f"trading_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2('trading_state.db', backup_file)
                st.success(f"Database backed up to {backup_file}")
            except Exception as e:
                st.error(f"Backup failed: {str(e)}")
        
        if st.button("Clear Cache"):
            try:
                if os.path.exists('market_cache.db'):
                    conn = sqlite3.connect('market_cache.db')
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM market_cache")
                    cursor.execute("DELETE FROM ohlcv_cache")
                    conn.commit()
                    conn.close()
                    st.cache_data.clear()
                    st.success("Cache cleared successfully")
            except Exception as e:
                st.error(f"Failed to clear cache: {str(e)}")

# Update timestamp in the footer
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
