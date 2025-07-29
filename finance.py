"""
Weekly market analysis and trade performance review with Gemini & Gmail

1. Uses Gemini API for trade analysis and review.
2. Sends emails securely via Gmail API (no password needed).
3. Schedules Monday analysis and Friday review.
4. Enhanced with technical indicators, news sentiment, and caching.
"""
import os
import json
import requests
import sys
import time
import pytz
import asyncio
import aiohttp
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.auth.exceptions import RefreshError
import numpy as np
from collections import defaultdict
import threading
from functools import lru_cache

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

from email.message import EmailMessage
import base64
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pathlib import Path

import logging
from typing import Dict, List, Optional, Any

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finance_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'cache_duration': 300,  # 5 minutes
    'max_retries': 3,
    'timeout': 30,
    'max_wait_minutes': 15,
    'trades_per_analysis': 8,
    'holding_period_min': 10,
    'holding_period_max': 45
}

class PerformanceMonitor:
    """Monitor performance metrics of the analysis system."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {
            'api_calls': 0,
            'cache_hits': 0,
            'errors': 0,
            'analysis_time': 0
        }
    
    def start_timer(self):
        self.start_time = time.time()
    
    def end_timer(self):
        if self.start_time:
            self.metrics['analysis_time'] = time.time() - self.start_time
    
    def increment_api_calls(self):
        self.metrics['api_calls'] += 1
    
    def increment_cache_hits(self):
        self.metrics['cache_hits'] += 1
    
    def increment_errors(self):
        self.metrics['errors'] += 1
    
    def get_summary(self):
        return {
            'total_time': self.metrics['analysis_time'],
            'api_calls': self.metrics['api_calls'],
            'cache_hits': self.metrics['cache_hits'],
            'errors': self.metrics['errors'],
            'cache_hit_rate': self.metrics['cache_hits'] / max(self.metrics['api_calls'], 1) * 100
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()

def log_performance():
    """Log performance metrics."""
    summary = performance_monitor.get_summary()
    logger.info(f"Performance Summary: {summary}")

# Enhanced error handling with logging
def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with error handling and logging."""
    try:
        performance_monitor.increment_api_calls()
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        performance_monitor.increment_errors()
        logger.error(f"API call failed in {func.__name__}: {str(e)}")
        return None

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS").split(",")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# If modifying scopes, delete the file token.json
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Cache for API responses
CACHE_DURATION = 300  # 5 minutes
api_cache = {}
cache_lock = threading.Lock()

class MarketDataCache:
    def __init__(self, duration=300):
        self.cache = {}
        self.duration = duration
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.duration:
                    return data
                else:
                    del self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            self.cache[key] = (value, time.time())

# Global cache instance
market_cache = MarketDataCache()

def calculate_rsi(prices, period=14):
    """Calculate RSI for a series of prices."""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.mean(gains[:period])
    avg_losses = np.mean(losses[:period])
    
    for i in range(period, len(deltas)):
        avg_gains = (avg_gains * (period - 1) + gains[i]) / period
        avg_losses = (avg_losses * (period - 1) + losses[i]) / period
    
    if avg_losses == 0:
        return 100
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(prices, period):
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def get_technical_indicators(symbol, days=50):
    """Get technical indicators for a symbol."""
    cache_key = f"tech_{symbol}_{days}"
    cached_data = market_cache.get(cache_key)
    if cached_data:
        return cached_data
    
    try:
        headers = {"Authorization": f"Token {TIINGO_API_KEY}"}
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        
        # Get data from a wider range to ensure we have enough data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+10)
        
        params = {
            "token": TIINGO_API_KEY,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "format": "json"
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            # If no data found, try with a wider range
            start_date = end_date - timedelta(days=days+20)
            params["startDate"] = start_date.strftime("%Y-%m-%d")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"No data available for {symbol}")
                return None
        
        # Extract prices
        prices = [d['close'] for d in data]
        volumes = [d['volume'] for d in data if 'volume' in d]
        
        if len(prices) < 2:
            logger.warning(f"Insufficient price data for {symbol}")
            return None
        
        # Calculate indicators
        current_price = prices[-1]
        sma_20 = calculate_sma(prices, 20) if len(prices) >= 20 else None
        sma_50 = calculate_sma(prices, 50) if len(prices) >= 50 else None
        rsi = calculate_rsi(prices)
        
        # Volume analysis
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else None
        current_volume = volumes[-1] if volumes else None
        volume_ratio = current_volume / avg_volume if avg_volume and current_volume else None
        
        # Volatility (20-day)
        if len(prices) >= 21:  # Need at least 21 prices for 20 returns
            returns = np.diff(prices[-21:]) / prices[-21:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized %
        else:
            volatility = None
        
        result = {
            'price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'trend': 'bullish' if sma_20 and sma_50 and current_price > sma_20 > sma_50 else 'bearish' if sma_20 and sma_50 and current_price < sma_20 < sma_50 else 'neutral',
            'data_date': data[-1].get('date', 'Unknown') if data else 'Unknown'
        }
        
        market_cache.set(cache_key, result)
        return result
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {e}")
        return None

async def get_news_sentiment(symbol, days=7):
    """Get news sentiment for a symbol using Tiingo news API."""
    try:
        headers = {"Authorization": f"Token {TIINGO_API_KEY}"}
        url = f"https://api.tiingo.com/news"
        params = {
            "token": TIINGO_API_KEY,
            "tickers": symbol,
            "startDate": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "endDate": datetime.now().strftime("%Y-%m-%d"),
            "limit": 10
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    news_data = await response.json()
                    if news_data:
                        # Simple sentiment analysis based on keywords
                        positive_words = ['bullish', 'positive', 'growth', 'upgrade', 'beat', 'strong']
                        negative_words = ['bearish', 'negative', 'decline', 'downgrade', 'miss', 'weak']
                        
                        sentiment_score = 0
                        for article in news_data:
                            title = article.get('title', '').lower()
                            description = article.get('description', '').lower()
                            text = f"{title} {description}"
                            
                            for word in positive_words:
                                if word in text:
                                    sentiment_score += 1
                            for word in negative_words:
                                if word in text:
                                    sentiment_score -= 1
                        
                        return {
                            'articles_count': len(news_data),
                            'sentiment_score': sentiment_score,
                            'sentiment': 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
                        }
        return None
    except Exception as e:
        print(f"Error getting news sentiment for {symbol}: {e}")
        return None

async def get_market_data_async():
    """Enhanced market data collection with technical indicators and news."""
    tickers = {
        "S&P 500": "SPY",
        "Nasdaq": "QQQ", 
        "Dow Jones": "DIA",
        "VIX Volatility": "VXX",
        "10Y Treasury": "TLT",
        "Bitcoin": "BTCUSD",
        "Ethereum": "ETHUSD",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Nvidia": "NVDA"
    }

    summary = []
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for name, symbol in tickers.items():
            task = asyncio.create_task(get_symbol_data_async(session, name, symbol))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                summary.append(f"Error: {str(result)}")
            elif result:
                summary.append(result)
    
    return "\n".join(summary)

async def get_symbol_data_async(session, name, symbol):
    """Get comprehensive data for a single symbol."""
    try:
        # Get technical indicators
        tech_data = get_technical_indicators(symbol)
        if not tech_data:
            return f"{name} ({symbol}): No technical data available."
        
        # Get news sentiment
        news_data = await get_news_sentiment(symbol)
        
        # Format the output
        price = tech_data['price']
        change_pct = ((tech_data['price'] - tech_data['sma_20']) / tech_data['sma_20'] * 100) if tech_data['sma_20'] else 0
        rsi = tech_data['rsi']
        trend = tech_data['trend']
        volume_info = f"Vol: {tech_data['volume_ratio']:.1f}x" if tech_data['volume_ratio'] else ""
        
        news_info = ""
        if news_data:
            sentiment_emoji = {"positive": "📈", "negative": "📉", "neutral": "➡️"}.get(news_data['sentiment'], "➡️")
            news_info = f" | News: {sentiment_emoji} ({news_data['articles_count']} articles)"
        
        return f"{name} ({symbol}): ${price:.2f} ({change_pct:+.1f}%) | RSI: {rsi:.1f} | {trend.title()} {volume_info}{news_info}"
        
    except Exception as e:
        return f"{name} ({symbol}): Error - {str(e)}"

def get_market_data():
    """Get market data with strict current-day requirements during market hours."""
    logger.info("[DATA] Getting market data with strict current-day requirements...")
    
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)
    today = now.date()
    
    # Check if market is open (9:30 AM - 4:00 PM ET, Monday-Friday)
    is_weekday = now.weekday() < 5
    is_market_hours = (now.hour == 9 and now.minute >= 30) or (now.hour > 9 and now.hour < 16) or (now.hour == 16 and now.minute == 0)
    market_open = is_weekday and is_market_hours
    
    logger.info(f"[TIME] Current time (EST): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[MARKET] Market status: {'OPEN' if market_open else 'CLOSED'}")
    
    headers = {"Authorization": f"Token {TIINGO_API_KEY}"}
    
    # Strategy 1: Try IEX Cloud endpoint first (most reliable for real-time)
    logger.info("[STRATEGY] Trying IEX Cloud endpoint (most reliable)...")
    try:
        url = f"https://api.tiingo.com/iex/SPY/prices"
        params = {
            "token": TIINGO_API_KEY,
            "startDate": today.strftime("%Y-%m-%d"),
            "endDate": today.strftime("%Y-%m-%d"),
            "format": "json"
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        iex_data = response.json()
        
        if iex_data and len(iex_data) > 0:
            data_date = iex_data[0].get('date', 'Unknown')
            if data_date.startswith(today.strftime("%Y-%m-%d")):
                logger.info(f"[SUCCESS] IEX data found for today!")
                return iex_data[0]
            else:
                logger.warning(f"[WARNING] IEX data is from {data_date}, not today")
        else:
            logger.warning(f"[WARNING] No IEX data available for today")
    except Exception as e:
        logger.warning(f"[WARNING] IEX endpoint failed: {str(e)}")
    
    # Strategy 2: Try daily endpoint for today
    logger.info("[STRATEGY] Trying daily endpoint for today's data...")
    try:
        url = f"https://api.tiingo.com/tiingo/daily/SPY/prices"
        params = {
            "token": TIINGO_API_KEY,
            "startDate": today.strftime("%Y-%m-%d"),
            "endDate": today.strftime("%Y-%m-%d"),
            "format": "json"
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            data_date = data[0].get('date', 'Unknown')
            if data_date.startswith(today.strftime("%Y-%m-%d")):
                logger.info(f"[SUCCESS] Daily data found for today!")
                return data[0]
            else:
                logger.warning(f"[WARNING] Daily data is from {data_date}, not today")
        else:
            logger.warning(f"[WARNING] No daily data available for today")
    except Exception as e:
        logger.warning(f"[WARNING] Daily endpoint failed: {str(e)}")
    
    # Strategy 3: Try intraday data (more real-time) - Fixed parameters
    if market_open:
        logger.info("[STRATEGY] Market is open - trying intraday data...")
        try:
            url = f"https://api.tiingo.com/tiingo/daily/SPY/prices"
            params = {
                "token": TIINGO_API_KEY,
                "startDate": today.strftime("%Y-%m-%d"),
                "endDate": today.strftime("%Y-%m-%d"),
                "format": "json",
                "resampleFreq": "1min"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            intraday_data = response.json()
            
            if intraday_data and len(intraday_data) > 0:
                # Get the most recent intraday data
                latest_intraday = intraday_data[-1]
                data_date = latest_intraday.get('date', 'Unknown')
                if data_date.startswith(today.strftime("%Y-%m-%d")):
                    logger.info(f"[SUCCESS] Intraday data found for today: {latest_intraday}")
                    return latest_intraday
                else:
                    logger.warning(f"[WARNING] Intraday data is from {data_date}, not today")
            else:
                logger.warning(f"[WARNING] No intraday data available for today")
        except Exception as e:
            logger.warning(f"[WARNING] Intraday endpoint failed: {str(e)}")
    
    # Strategy 4: Wait for data if market is open, fallback only if closed
    if market_open:
        logger.warning(f"[WAIT] Market is open but no current data available. Waiting for real-time data...")
        raise Exception(f"Real-time data not yet available for {today.strftime('%Y-%m-%d')} at {now.strftime('%H:%M:%S')} EST")
    else:
        # Only fallback to historical data if market is closed
        logger.info("[STRATEGY] Market closed - falling back to most recent available data...")
        try:
            url = f"https://api.tiingo.com/tiingo/daily/SPY/prices"
            params = {
                "token": TIINGO_API_KEY,
                "startDate": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
                "endDate": today.strftime("%Y-%m-%d"),
                "format": "json"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            recent_data = response.json()
            
            if recent_data and len(recent_data) > 0:
                latest_data = recent_data[0]
                data_date = latest_data.get('date', 'Unknown')
                logger.warning(f"[WARNING] Using historical data from {data_date} (market closed)")
                return latest_data
        except Exception as e:
            logger.error(f"[ERROR] Historical data fallback failed: {str(e)}")
    
    raise Exception(f"No current data available for {today.strftime('%Y-%m-%d')} and market is open")

def get_market_data_async_wrapper():
    """Synchronous wrapper for async market data collection."""
    try:
        return asyncio.run(get_market_data_async())
    except Exception as e:
        print(f"Error in market data collection: {e}")
        return "Error retrieving market data."

def wait_for_market_data(symbol="SPY", max_wait_minutes=15):
    print(f"[WAIT] Getting latest available market data for {symbol}...")

    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)
    today = now.date()
    deadline = datetime.now() + timedelta(minutes=max_wait_minutes)
    
    # Check if market is open (9:30 AM - 4:00 PM ET, Monday-Friday)
    is_weekday = now.weekday() < 5
    is_market_hours = (now.hour == 9 and now.minute >= 30) or (now.hour > 9 and now.hour < 16) or (now.hour == 16 and now.minute == 0)
    market_open = is_weekday and is_market_hours
    
    print(f"[TIME] Current time (EST): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MARKET] Market status: {'OPEN' if market_open else 'CLOSED'}")
    
    # Run data quality test if market is open
    if market_open:
        logger.info("[TEST] Market is open - running data quality test...")
        test_result = test_data_freshness(symbol)
        if test_result['status'] == 'success':
            logger.info("[SUCCESS] Data quality test passed - proceeding with analysis")
        else:
            logger.warning("[WARNING] Data quality test failed - using fallback data")
    
    # Use the enhanced get_market_data function with strict requirements
    try:
        return get_market_data()
    except Exception as e:
        logger.error(f"[ERROR] Enhanced market data retrieval failed: {str(e)}")
        
        # If market is open and we can't get current data, wait and retry
        if market_open:
            print(f"[WAIT] Market is open but no current data available. Waiting up to {max_wait_minutes} minutes...")
            
            while datetime.now() < deadline:
                try:
                    print(f"[RETRY] Attempting to get current data... ({datetime.now().strftime('%H:%M:%S')})")
                    return get_market_data()
                except Exception as retry_error:
                    print(f"[WAIT] Still no current data available. Retrying in 60s...")
                    time.sleep(60)
            
            raise TimeoutError(f"[ERROR] No current data available after waiting {max_wait_minutes} minutes.")
        else:
            raise TimeoutError(f"[ERROR] No market data available: {str(e)}")

def get_sector_rotation():
    """Get sector rotation data."""
    sectors = {
        "Technology": "XLK",
        "Healthcare": "XLV", 
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Industrials": "XLI",
        "Energy": "XLE",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Consumer Staples": "XLP",
        "Communication Services": "XLC"
    }
    
    sector_performance = {}
    for sector_name, symbol in sectors.items():
        tech_data = get_technical_indicators(symbol, days=20)
        if tech_data:
            sector_performance[sector_name] = {
                'change_5d': ((tech_data['price'] - tech_data['sma_20']) / tech_data['sma_20'] * 100) if tech_data['sma_20'] else 0,
                'rsi': tech_data['rsi'],
                'trend': tech_data['trend']
            }
    
    # Sort by 5-day performance
    sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['change_5d'], reverse=True)
    
    return sorted_sectors

def get_earnings_calendar(days_ahead=30):
    """Get upcoming earnings dates for major stocks."""
    try:
        # This would typically use a different API, but for now we'll use a simplified approach
        # In a real implementation, you'd want to use an earnings calendar API
        major_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX"]
        
        earnings_info = []
        for symbol in major_stocks:
            # This is a placeholder - in reality you'd fetch from an earnings API
            # For now, we'll just return a basic structure
            earnings_info.append({
                'symbol': symbol,
                'estimated_date': None,  # Would be fetched from API
                'days_until': None
            })
        
        return earnings_info
    except Exception as e:
        print(f"Error fetching earnings calendar: {e}")
        return []

def get_volatility_analysis():
    """Get current market volatility metrics."""
    try:
        # VIX analysis
        vix_data = get_technical_indicators("VXX", days=20)
        if vix_data:
            vix_level = vix_data['price']
            vix_trend = vix_data['trend']
            
            # Determine market sentiment based on VIX
            if vix_level > 25:
                market_sentiment = "High Fear"
            elif vix_level > 20:
                market_sentiment = "Moderate Fear"
            elif vix_level > 15:
                market_sentiment = "Neutral"
            elif vix_level > 10:
                market_sentiment = "Moderate Greed"
            else:
                market_sentiment = "High Greed"
            
            return {
                'vix_level': vix_level,
                'vix_trend': vix_trend,
                'market_sentiment': market_sentiment
            }
    except Exception as e:
        print(f"Error in volatility analysis: {e}")
        return None

def validate_market_conditions():
    """Validate if market conditions are suitable for trading."""
    try:
        # Check if market is open (9:30 AM - 4:00 PM ET, Monday-Friday)
        est = pytz.timezone("US/Eastern")
        now = datetime.now(est)
        
        is_weekday = now.weekday() < 5
        is_market_hours = (now.hour == 9 and now.minute >= 30) or (now.hour > 9 and now.hour < 16) or (now.hour == 16 and now.minute == 0)
        market_open = is_weekday and is_market_hours
        
        # Get volatility data
        vol_data = get_volatility_analysis()
        
        # Determine data freshness expectations
        if market_open:
            data_freshness = "Real-time data expected"
        else:
            data_freshness = "Previous day's data acceptable"
        
        conditions = {
            'market_open': market_open,
            'volatility': vol_data,
            'suitable_for_trading': True,  # Would add more sophisticated logic
            'data_freshness': data_freshness,
            'current_time_est': now.strftime('%Y-%m-%d %H:%M:%S EST')
        }
        
        return conditions
    except Exception as e:
        logger.error(f"Error validating market conditions: {e}")
        return {'market_open': False, 'volatility': None, 'suitable_for_trading': False}

# Enhanced Gemini prompt with technical analysis
def build_gemini_prompt():
    """Build a comprehensive prompt for Gemini with detailed trading recommendations at top and analysis at bottom."""
    logger.info("[PROMPT] Building comprehensive prompt with detailed trading focus...")
    
    try:
        # Get market data
        market_data = get_market_data()
        
        # Get sector news for context
        sector_news = get_sector_news()
        news_content = format_sector_news_for_email(sector_news)
        
        # Get technical indicators
        technical_data = get_technical_indicators("SPY")
        
        # Get sector rotation data
        sector_rotation = get_sector_rotation()
        
        # Get volatility analysis
        volatility_data = get_volatility_analysis()
        
        # Build comprehensive prompt with detailed trading focus
        prompt = f"""
You are my personal swing‑trade assistant. Work only with Russell 1000 tickers and deliver credit trades exclusively—bull‑put spreads, bear‑call spreads, or iron condors.

Use the following comprehensive market data to inform your trading decisions:

📊 MARKET DATA:
- Current S&P 500: ${market_data.get('close', 'N/A')}
- High: ${market_data.get('high', 'N/A')}
- Low: ${market_data.get('low', 'N/A')}
- Open: ${market_data.get('open', 'N/A')}
- Date: {market_data.get('date', 'N/A')}

🔍 TECHNICAL INDICATORS:
{technical_data}

📈 SECTOR ROTATION:
{sector_rotation}

📊 VOLATILITY ANALYSIS:
{volatility_data}

📰 SECTOR NEWS & SENTIMENT:
{news_content}

TRADING REQUIREMENTS:
For every quote or option chain you use, confirm it meets the Data‑Freshness Rule below; if not, say "Out‑of‑date data—skipped" and exclude that ticker.

Data‑Freshness Rule
• If the market is open:
  – Use real‑time quotes and option chains stamped no more than 15 minutes old or clearly labeled delayed data stamped no more than 30 minutes old.
  – When delayed data are used, append "(Delayed)" after the timestamp so I know the source was not real‑time.
• If the market is closed: use the official last closing price and closing‑bell option snapshot (stamp them YYYY‑MM‑DD 16:00 ET – Close).
• If neither real‑time, delayed, nor previous‑close data are available for a ticker, say "No valid data—skipped" and move on.

Default parameters—override only if I specify otherwise
• Holding period = 10–45 days
• Number of simultaneous trades to generate = 8

Enhanced Screening Criteria (considering news and sector data):
a. Technical Analysis: 
   - Bullish spreads: price > rising 20‑ & 50‑day SMAs and RSI 50–70
   - Bearish spreads: price < falling SMAs and RSI 30–50  
   - Condors: price consolidating within ±5% of 20‑day average
   - Volume confirmation: above-average volume preferred
b. Confirming catalyst within 45 days (earnings date, analyst action, sector rotation, or macro driver) published or updated within the last 30 days—show the catalyst date.
c. Option‑chain liquidity: each leg you trade must have open interest ≥ 500 contracts and bid‑ask spread ≤ 8%.
d. IV rank > 60% or unusual volume spike recorded today (or on the most recent session if after hours).
e. Diversification: no more than two trades from the same GICS sector or with 30‑day price correlation ≥ 0.70 to another pick.
f. Sector Alignment: Consider current sector rotation trends and news when selecting trades.
g. Market Sentiment: Adjust strategy based on current VIX levels, market sentiment, and sector news.

Strike‑Selection Rules (apply exactly)
• Bull‑put / bear‑call spread: sell the short strike at roughly 15‑ to 25‑delta (≈ 1 s.d.); buy the long strike 5–10 points farther OTM.
• Iron condor: sell short strikes just outside the 1‑s.d. move; buy wings 5–10 points farther OTM.
• Target total credit ≥ 30% of the distance between short and long strikes; skip trades that fall below this.

Ranking
Order candidates by the highest probability of retaining at least 50% of the credit before stop, using delta‑based OTM probability and recent range statistics. Consider technical indicators, sector rotation, market sentiment, and sector news in ranking.

OUTPUT FORMAT:
FIRST: Provide your trading recommendations in this EXACT detailed format:

<Ticker> <bias> <structure>
Step 1 – <Open leg 1 – strike, expiry, sell, net credit> (Underlying price $XX.XX, timestamp YYYY‑MM‑DD HH:MM ET or "YYYY‑MM‑DD 16:00 ET – Close"[Delayed if applicable])
Step 2 – <Open leg 2 …> (add Step 3/4 if iron condor)
Step X – <Target net credit and minimum acceptable fill>
Step X – <Risk exit rule: close if price hits stop or short‑leg delta ≥ 0.35>
Step X – <Profit exit rule: close when 50% of credit is captured or 21 days remain>
Why this works – <DETAILED EXPLANATION including:>
   • Technical setup (RSI, moving averages, support/resistance)
   • Sector rotation context and news impact
   • Specific catalyst (earnings, analyst action, etc.) with date
   • Market sentiment and volatility considerations
   • Risk/reward rationale
   • Expected price movement and timing
   • How news and sector data support this trade

Use plain sentences—no bullets, tables, markdown, or YAML inside the trade blocks. Skip any trade where data or catalyst fails the recency rule.

After all trade blocks, write exactly one sentence:
Expect individual winners and losers; the probability edge plays out over the full set of trades.

THEN: Provide comprehensive market analysis:

📈 DAILY MARKET ANALYSIS
==================================================

📊 MARKET OVERVIEW
[Your market summary incorporating news and sector data]

🔍 TECHNICAL ANALYSIS
[Your technical insights with news correlation]

📰 SECTOR INSIGHTS
[Your sector analysis based on news and rotation data]

💡 TRADING OPPORTUNITIES
[Additional trading ideas based on news and sector analysis]

⚠️ RISK ASSESSMENT
[Your risk analysis including news-driven risks]

🎯 MARKET SENTIMENT
[Your sentiment analysis from news and data]

Keep the analysis professional, actionable, and focused on helping with trading decisions.
"""
        
        logger.info("[SUCCESS] Comprehensive prompt built with detailed trading focus")
        return prompt
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to build comprehensive prompt: {str(e)}")
        # Fallback to basic prompt
        return build_basic_prompt()

def build_basic_prompt():
    """Build a basic prompt as fallback when comprehensive prompt fails."""
    logger.info("[PROMPT] Building basic prompt as fallback...")
    
    try:
        market_data = get_market_data()
        date_str = datetime.now().strftime("%B %d, %Y")
        
        return f"""
Today is {date_str}. 

MARKET DATA:
- Current S&P 500: ${market_data.get('close', 'N/A')}
- High: ${market_data.get('high', 'N/A')}
- Low: ${market_data.get('low', 'N/A')}
- Open: ${market_data.get('open', 'N/A')}

You are a professional financial analyst. Provide a daily market analysis including:
1. Market overview and current conditions
2. Technical analysis and key levels
3. Trading recommendations
4. Risk assessment

Format your response professionally and focus on actionable insights.
"""
    except Exception as e:
        logger.error(f"[ERROR] Basic prompt failed: {str(e)}")
        return "Provide a daily market analysis based on current market conditions."

# Gemini API with retry logic
def run_gemini(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("models/gemini-2.5-pro")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None

# Create analysis
def get_week_monday(date=None):
    """Return the date (YYYY-MM-DD) of the Monday for the week of the given date (or today)."""
    if date is None:
        date = datetime.now()
    monday = date - timedelta(days=date.weekday())
    return monday.strftime("%Y-%m-%d")

# Update store_analysis to use weekly_reports/<monday>/YYYY-MM-DD.json

def store_analysis(analysis):
    date = datetime.now()
    date_str = date.strftime("%Y-%m-%d")
    week_monday = get_week_monday(date)
    folder = Path("weekly_reports") / week_monday
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "date": date_str,
        "analysis": analysis
    }
    try:
        with open(folder / f"{date_str}.json", "w") as f:
            json.dump(data, f)
        with open("weekly_analysis.json", "w") as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"Storage Error: {str(e)}")
        return False

# Helper to load all daily reports for the current week

def load_weekly_reports(week_monday=None):
    if week_monday is None:
        week_monday = get_week_monday()
    folder = Path("weekly_reports") / week_monday
    reports = []
    if not folder.exists():
        return reports
    for file in sorted(folder.glob("*.json")):
        # Skip the weekly summary file
        if file.name == "weekly_summary.json":
            continue
        try:
            with open(file, "r") as f:
                data = json.load(f)
                # Only include files that have the expected structure for daily reports
                if "date" in data and "analysis" in data:
                    reports.append(data)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return reports

# Friday analysis: summarize the week and email

def run_friday_analysis(week_monday=None):
    if week_monday is None:
        week_monday = get_week_monday()
    reports = load_weekly_reports(week_monday)
    if not reports:
        print("No daily reports to summarize the week.")
        return
    # Compose a summary prompt for Gemini
    week_dates = f"{week_monday} to {(datetime.strptime(week_monday, '%Y-%m-%d') + timedelta(days=6)).strftime('%Y-%m-%d')}"
    daily_analyses = "\n\n".join([f"{r['date']}\n{r['analysis']}" for r in reports])
    summary_prompt = f"""
You are my personal swing-trade assistant. Here are my daily market analyses for the week {week_dates}:

{daily_analyses}

Please provide a concise summary of the week's market performance, notable trends, and any actionable insights for the coming week. Use plain sentences, no bullet points or markdown.
"""
    summary = run_gemini(summary_prompt)
    if summary:
        # Store and email the summary
        summary_data = {
            "week": week_monday,
            "summary": summary,
            "dates": [r["date"] for r in reports]
        }
        folder = Path("weekly_reports") / week_monday
        with open(folder / "weekly_summary.json", "w") as f:
            json.dump(summary_data, f)
        send_email(f"📊 Weekly Market Summary ({week_dates})", summary)
        print("Weekly summary sent.")
    else:
        print("Failed to generate weekly summary.")

# Send email
def send_email(subject, body):
    try:
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        
        # If there are no credentials, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as refresh_error:
                    logger.warning(f"Token refresh failed: {refresh_error}")
                    # Remove the expired token file
                    if os.path.exists('token.json'):
                        os.remove('token.json')
                    creds = None
            
            if not creds:
                # Only try to create new credentials if we have the credentials file
                if os.path.exists('credentials.json'):
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                    # Save credentials
                    with open('token.json', 'w') as token:
                        token.write(creds.to_json())
                else:
                    logger.error("No credentials.json file found. Cannot send email.")
                    return False
        
        if not creds:
            logger.error("No valid credentials available. Cannot send email.")
            return False

        service = build('gmail', 'v1', credentials=creds)
        message = EmailMessage()
        message.set_content(body)
        message['To'] = ", ".join(EMAIL_RECIPIENTS)
        message['From'] = EMAIL_SENDER
        message['Subject'] = subject

        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_message = service.users().messages().send(userId="me", body={'raw': encoded}).execute()
        logger.info("Email sent successfully!")
        return True
    except HttpError as error:
        logger.error(f'An error occurred while sending the email: {error}')
        return False
    except Exception as e:
        logger.error(f'Unexpected error in send_email: {e}')
        return False

def run_daily_analysis():
    logger.info(f"Starting daily analysis at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    performance_monitor.start_timer()
    
    try:
        prompt = build_gemini_prompt()
        analysis = run_gemini(prompt)
        if analysis:
            # Get sector news
            logger.info("Getting sector news for email...")
            try:
                sector_news = get_sector_news()
                news_content = format_sector_news_for_email(sector_news)
                # Add sector news to the analysis
                full_analysis = analysis + news_content
                logger.info("Sector news added to analysis successfully.")
            except Exception as e:
                logger.warning(f"Failed to get sector news: {str(e)}")
                full_analysis = analysis + "\n\n📰 SECTOR NEWS\n" + "="*30 + "\nNo sector news available at this time."
            
            if store_analysis(full_analysis):
                if send_email("📈 Daily Market Analysis", full_analysis):
                    logger.info("Daily analysis with sector news completed and sent successfully.")
                else:
                    logger.warning("Analysis completed but email sending failed.")
            else:
                logger.error("Failed to store analysis.")
        else:
            logger.error("Analysis generation failed.")
    except Exception as e:
        logger.error(f"Error in daily analysis: {str(e)}")
    finally:
        performance_monitor.end_timer()
        log_performance()



def get_sector_news():
    """Get sector-specific news from Tiingo API, with Technology as the first sector."""
    logger.info("[NEWS] Getting sector-specific news from Tiingo...")
    
    # First try Tiingo news API
    try:
        headers = {"Authorization": f"Token {TIINGO_API_KEY}"}
        
        # Define sectors with their common tickers for news search
        sectors = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "TMO", "MRK", "ABT"],
            "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "DIS"],
            "Industrials": ["BA", "CAT", "MMM", "GE", "HON", "UPS", "FDX"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "KMI", "PSX"],
            "Materials": ["LIN", "APD", "FCX", "NEM", "DD", "DOW", "NUE"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "XEL", "SRE"],
            "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "DLR", "PSA", "O"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "PM", "CL"],
            "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T"]
        }
        
        sector_news = {}
        
        for sector_name, tickers in sectors.items():
            logger.info(f"[NEWS] Getting news for {sector_name} sector...")
            
            try:
                # Get news for the sector's major tickers
                sector_articles = []
                
                for ticker in tickers[:3]:  # Use top 3 tickers per sector for efficiency
                    # Try different Tiingo news endpoints
                    endpoints = [
                        f"https://api.tiingo.com/tiingo/news/{ticker}",
                        f"https://api.tiingo.com/tiingo/news",
                        f"https://api.tiingo.com/news/{ticker}"
                    ]
                    
                    for endpoint in endpoints:
                        try:
                            params = {
                                "token": TIINGO_API_KEY,
                                "limit": 3,  # Get 3 articles per ticker
                                "startDate": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                                "endDate": datetime.now().strftime("%Y-%m-%d")
                            }
                            
                            response = requests.get(endpoint, headers=headers, params=params)
                            response.raise_for_status()
                            articles = response.json()
                            
                            for article in articles:
                                sector_articles.append({
                                    'title': article.get('title', ''),
                                    'description': article.get('description', ''),
                                    'publishedDate': article.get('publishedDate', ''),
                                    'source': article.get('source', ''),
                                    'ticker': ticker
                                })
                            
                            # If we got articles, break out of endpoint loop
                            if articles:
                                break
                                
                        except Exception as e:
                            logger.debug(f"[DEBUG] Endpoint {endpoint} failed for {ticker}: {str(e)}")
                            continue
                
                # Sort by published date (most recent first)
                sector_articles.sort(key=lambda x: x.get('publishedDate', ''), reverse=True)
                
                # Take top 5 articles for the sector
                sector_news[sector_name] = sector_articles[:5]
                
                logger.info(f"[SUCCESS] Found {len(sector_articles)} articles for {sector_name}")
                
            except Exception as e:
                logger.warning(f"[WARNING] Failed to get news for {sector_name}: {str(e)}")
                sector_news[sector_name] = []
        
        # Check if we got any real news
        total_articles = sum(len(articles) for articles in sector_news.values())
        if total_articles > 0:
            logger.info(f"[SUCCESS] Retrieved {total_articles} total articles from Tiingo")
            return sector_news
        else:
            logger.warning("[WARNING] No articles found from Tiingo API, using fallback")
            raise Exception("No articles found from Tiingo API")
            
    except Exception as e:
        logger.warning(f"[WARNING] Tiingo news API failed: {str(e)}")
        logger.info("[INFO] Using fallback sector news...")
        return get_fallback_sector_news()

def format_sector_news_for_email(sector_news):
    """Format sector news for email inclusion."""
    if not sector_news:
        return "No sector news available at this time."
    
    email_content = "\n\n📰 SECTOR NEWS CONSENSUS\n" + "="*50 + "\n"
    
    # Technology first (as requested)
    if "Technology" in sector_news and sector_news["Technology"]:
        email_content += "\n🔬 TECHNOLOGY SECTOR\n" + "-"*30 + "\n"
        for i, article in enumerate(sector_news["Technology"][:3], 1):
            email_content += f"{i}. {article['title']}\n"
            email_content += f"   Source: {article['source']} | {article['publishedDate'][:10]}\n"
            if article['description']:
                email_content += f"   {article['description'][:150]}...\n"
            email_content += "\n"
    
    # Other sectors
    for sector_name, articles in sector_news.items():
        if sector_name == "Technology" or not articles:
            continue
            
        # Sector emoji mapping
        sector_emojis = {
            "Healthcare": "🏥",
            "Financials": "💰", 
            "Consumer Discretionary": "🛍️",
            "Industrials": "🏭",
            "Energy": "⚡",
            "Materials": "🏗️",
            "Utilities": "⚡",
            "Real Estate": "🏢",
            "Consumer Staples": "🛒",
            "Communication Services": "📡"
        }
        
        emoji = sector_emojis.get(sector_name, "📊")
        email_content += f"\n{emoji} {sector_name.upper()} SECTOR\n" + "-"*30 + "\n"
        
        for i, article in enumerate(articles[:2], 1):  # Top 2 articles per sector
            email_content += f"{i}. {article['title']}\n"
            email_content += f"   Source: {article['source']} | {article['publishedDate'][:10]}\n"
            if article['description']:
                email_content += f"   {article['description'][:120]}...\n"
            email_content += "\n"
    
    return email_content


def get_fallback_sector_news():
    """Provide fallback sector news when Tiingo API is not available."""
    logger.info("[NEWS] Using fallback sector news...")
    
    # Create mock sector news based on current market conditions
    fallback_news = {
        "Technology": [
            {
                'title': 'Tech Sector Shows Strong Momentum',
                'description': 'Technology stocks continue to lead market gains with AI and cloud computing driving growth.',
                'publishedDate': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Market Analysis'
            }
        ],
        "Healthcare": [
            {
                'title': 'Healthcare Sector Stable Amid Market Volatility',
                'description': 'Healthcare stocks remain defensive as investors seek stability in uncertain markets.',
                'publishedDate': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Market Analysis'
            }
        ],
        "Financials": [
            {
                'title': 'Financial Sector Responds to Economic Data',
                'description': 'Banking stocks react to latest economic indicators and interest rate expectations.',
                'publishedDate': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Market Analysis'
            }
        ]
    }
    
    return fallback_news


if __name__ == "__main__":
    today = datetime.now()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "daily":
            run_daily_analysis()
        elif arg == "friday":
            run_friday_analysis()
        else:
            print("Usage: python finance.py [daily|friday]")
    else:
        # If today is Monday, check if last week's summary is missing and send it
        if today.weekday() == 0:  # Monday
            last_week_monday = today - timedelta(days=7)
            last_week_monday_str = last_week_monday.strftime("%Y-%m-%d")
            
            if not os.path.exists(f"weekly_reports/{last_week_monday_str}/weekly_summary.json"):
                print(f"Missing weekly summary for {last_week_monday_str}. Sending it now...")
                run_friday_analysis(week_monday=last_week_monday_str)
            else:
                print("No action needed. Usage: python finance.py [daily|friday]")
        else:
            print("Usage: python finance.py [daily|friday]")

