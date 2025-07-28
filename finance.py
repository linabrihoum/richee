"""
Weekly market analysis and trade performance review with Gemini & Gmail

1. Uses Gemini API for trade analysis and review.
2. Sends emails securely via Gmail API (no password needed).
3. Schedules Monday analysis and Friday review.
"""
import os
import json
import requests
import sys
import time
import pytz
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.auth.exceptions import RefreshError

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

from email.message import EmailMessage
import base64
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pathlib import Path

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

# Helper: Collect market data using Tiingo
def get_market_data():
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
    headers = {"Authorization": f"Token {TIINGO_API_KEY}"}
    
    for name, symbol in tickers.items():
        try:
            # Get latest price data
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
            params = {
                "token": TIINGO_API_KEY,
                "startDate": datetime.now().strftime("%Y-%m-%d"),
                "endDate": datetime.now().strftime("%Y-%m-%d"),
                "format": "json"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            if response.json():
                data = response.json()[0]
                price = data["close"]
                change = data["close"] - data["open"]
                pct = (change / data["open"]) * 100
                summary.append(f"{name} ({symbol}): {price:.2f} ({change:+.2f}, {pct:+.2f}%)")
            else:
                summary.append(f"{name} ({symbol}): No data available.")
        except Exception as e:
            summary.append(f"{name} ({symbol}): Error retrieving data - {str(e)}")
    
    return "\n".join(summary)


def wait_for_market_data(symbol="SPY", max_wait_minutes=15):
    print(f"⏳ Waiting for up-to-date market data on {symbol}...")

    est = pytz.timezone("US/Eastern")
    today = datetime.now(est).date()
    deadline = datetime.now() + timedelta(minutes=max_wait_minutes)
    
    headers = {"Authorization": f"Token {TIINGO_API_KEY}"}

    while datetime.now() < deadline:
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
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
                latest_data = data[0]
                print(f"✅ Market data found for {today} at {latest_data.get('date', 'N/A')}")
                return latest_data
            else:
                print(f"❌ Market data for {today} not yet available. Retrying in 60s...")
                time.sleep(60)
        except Exception as e:
            print(f"❌ Error fetching data for {today}: {str(e)}. Retrying in 60s...")
            time.sleep(60)

    raise TimeoutError(f"❌ Market data for {today} not available after waiting {max_wait_minutes} minutes.")

# Gemini prompt
def build_gemini_prompt():
    date_str = datetime.now().strftime("%B %d, %Y")
    market_summary = get_market_data()
    return f"""
Today is {date_str}. Here's the latest real-time financial market data:

{market_summary}

You are my personal swing‑trade assistant. Work only with Russell 1000 tickers and deliver credit trades exclusively—bull‑put spreads, bear‑call spreads, or iron condors.



Based on this data, 

For every quote or option chain you use, confirm it meets the Data‑Freshness Rule below; if not, say “Out‑of‑date data—skipped” and exclude that ticker.

Data‑Freshness Rule
• If the market is open:
  – Use real‑time quotes and option chains stamped no more than 15 minutes old or clearly labeled delayed data stamped no more than 30 minutes old.
  – When delayed data are used, append “(Delayed)” after the timestamp so I know the source was not real‑time.
• If the market is closed: use the official last closing price and closing‑bell option snapshot (stamp them YYYY‑MM‑DD 16:00 ET – Close).
• If neither real‑time, delayed, nor previous‑close data are available for a ticker, say “No valid data—skipped” and move on.

Default parameters—override only if I specify otherwise
• Holding period = 10–45 days
• Number of simultaneous trades to generate = 8

Screening Criteria
a. Trend & RSI: bullish spreads need price > rising 20‑ & 50‑day SMAs and RSI 50–70; bearish spreads need price < falling SMAs and RSI 30–50; condors need price consolidating within ±5 % of its 20‑day average.
b. Confirming catalyst within 45 days (earnings date, analyst action, sector rotation, or macro driver) published or updated within the last 30 days—show the catalyst date.
c. Option‑chain liquidity: each leg you trade must have open interest ≥ 500 contracts and bid‑ask spread ≤ 8 %.
d. IV rank > 60 % or unusual volume spike recorded today (or on the most recent session if after hours).
e. Diversification: no more than two trades from the same GICS sector or with 30‑day price correlation ≥ 0.70 to another pick.

Strike‑Selection Rules (apply exactly)
• Bull‑put / bear‑call spread: sell the short strike at roughly 15‑ to 25‑delta (≈ 1 s.d.); buy the long strike 5–10 points farther OTM.
• Iron condor: sell short strikes just outside the 1‑s.d. move; buy wings 5–10 points farther OTM.
• Target total credit ≥ 30 % of the distance between short and long strikes; skip trades that fall below this.

Ranking
Order candidates by the highest probability of retaining at least 50 % of the credit before stop, using delta‑based OTM probability and recent range statistics. Return exactly the number of trades requested.

Output Format
For each trade, output one block (leave one blank line between blocks, none inside):

<Ticker> <bias> <structure>
Step 1 – <Open leg 1 – strike, expiry, sell, net credit> (Underlying price $XX.XX, timestamp YYYY‑MM‑DD HH:MM ET or “YYYY‑MM‑DD 16:00 ET – Close”[Delayed if applicable])
Step 2 – <Open leg 2 …> (add Step 3/4 if iron condor)
Step X – <Target net credit and minimum acceptable fill>
Step X – <Risk exit rule: close if price hits stop or short‑leg delta ≥ 0.35>
Step X – <Profit exit rule: close when 50 % of credit is captured or 21 days remain>
Why this works – <one concise sentence naming the edge or catalyst with its date>.

Use plain sentences—no bullets, tables, markdown, or YAML inside the trade blocks. Skip any trade where data or catalyst fails the recency rule.

After all trade blocks, write exactly one sentence:
Expect individual winners and losers; the probability edge plays out over the full set of trades.


"""

# Gemini API
def run_gemini(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
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
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)
        message = EmailMessage()
        message.set_content(body)
        message['To'] = ", ".join(EMAIL_RECIPIENTS)
        message['From'] = EMAIL_SENDER
        message['Subject'] = subject

        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_message = service.users().messages().send(userId="me", body={'raw': encoded}).execute()
        print(f"Email sent successfully!")
    except HttpError as error:
        print(f'An error occurred while sending the email: {error}')

# Run finance analysis based on the market for current day
def run_daily_analysis():
    print(f"Running Daily analysis ({datetime.now().strftime('%Y-%m-%d %H:%M')})...")
    prompt = build_gemini_prompt()
    analysis = run_gemini(prompt)
    if analysis:
        if store_analysis(analysis):
            send_email("📈 Daily Market Analysis", analysis)
        else:
            print("Failed to store analysis.")
    else:
        print("Analysis generation failed.")

if __name__ == "__main__":
    # Wait for market data before continuing
    wait_for_market_data(symbol="SPY")

    today = datetime.now()
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "daily":
            run_daily_analysis()
        elif arg == "friday":
            run_friday_analysis()
        else:
            print("Usage: python finance.py [daily|friday]")
    else:
        # If today is Monday, check if last week's summary is missing and send it
        if today.weekday() == 0:  # Monday
            last_monday = today - timedelta(days=7)
            last_week_monday_str = get_week_monday(last_monday)
            last_week_folder = Path("weekly_reports") / last_week_monday_str
            summary_file = last_week_folder / "weekly_summary.json"
            if last_week_folder.exists() and not summary_file.exists():
                print("Previous week's summary missing. Sending now...")
                run_friday_analysis(week_monday=last_week_monday_str)
            else:
                print("No action needed. Usage: python finance.py [daily|friday]")
        else:
            print("Usage: python finance.py [daily|friday]")
