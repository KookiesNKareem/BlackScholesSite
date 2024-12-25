import datetime
import numpy as np
from scipy.stats import norm
import yfinance as yf
from flask import Flask, request, render_template, jsonify
from fredapi import Fred
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)


def black_scholes_call_price(S, K, T, r, sigma):
    """
    Calculate call option price using the Black-Scholes formula.

    :param S: Current stock price
    :param K: Strike price
    :param T: Time to expiration in years
    :param r: Risk-free interest rate (annual)
    :param sigma: Volatility (annual)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put_price(S, K, T, r, sigma):
    """
    Calculate put option price using the Black-Scholes formula.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def monte_carlo_option_pricing(S, K, T, r, sigma, option_type, simulations=10000):
    """
    Monte Carlo simulation for call/put option pricing using Geometric Brownian Motion.

    :param S: Current stock price
    :param K: Strike price
    :param T: Time to expiration (in years)
    :param r: Risk-free interest rate
    :param sigma: Volatility (annual)
    :param option_type: "call" or "put"
    :param simulations: Number of Monte Carlo simulations
    """
    # Generate random draws from a normal distribution
    rand = np.random.normal(0, 1, simulations)

    # Simulate stock price after T years
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * rand)

    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    # Discounted average payoff
    option_price_mc = np.exp(-r * T) * np.mean(payoffs)
    return option_price_mc


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/option_data', methods=['GET'])
def get_option_data():
    """
    API endpoint to:
      1. Fetch the current price from yfinance.
      2. Fetch next 3 option expiration dates + top ~5 strikes near current price for each date.
      3. Get risk-free rate from FRED or fallback.
      4. Return JSON.
    """
    symbol = request.args.get('symbol', '').upper().strip()
    print(symbol)
    if not symbol:
        return jsonify({"error": "No ticker symbol provided"}), 400

    # 1) Fetch current price
    try:
        ticker_data = yf.Ticker(symbol)
        hist = ticker_data.history(period='1d')
        if hist.empty:
            return jsonify({"error": f"Could not fetch price for {symbol}"}), 404
        current_price = float(hist['Close'].iloc[-1])
    except Exception as e:
        return jsonify({"error": f"Error fetching data for {symbol}: {str(e)}"}), 500

    # 2) Fetch the next 3 expiration dates
    try:
        all_expirations = ticker_data.options
    except Exception as e:
        all_expirations = []
        print(f"[WARN] Could not load option chain for {symbol}: {e}")

    next_expiries = all_expirations[:3] if all_expirations else []

    expiries_data = {}
    for expiry in next_expiries:
        try:
            chain = ticker_data.option_chain(expiry)

            # same strike list
            calls = chain.calls.copy()
            if "strike" not in calls.columns:
                expiries_data[expiry] = []
                continue

            # compute difference from current price and pick ~5 closest
            calls["diff"] = (calls["strike"] - current_price).abs()
            calls_sorted = calls.sort_values("diff")
            closest_strikes = calls_sorted["strike"].head(5).tolist()

            closest_strikes = sorted(closest_strikes)

            expiries_data[expiry] = closest_strikes
        except Exception as e:
            print(f"[WARN] Could not fetch chain for {symbol} {expiry}: {e}")
            expiries_data[expiry] = []

    # 3) Get the risk-free rate from FRED
    fred_api_key = os.environ.get('FRED_API_KEY', '')
    if not fred_api_key or Fred is None:
        # fallback if no API key or FRED api not installed
        risk_free_rate = 3.0
    else:
        try:
            fred = Fred(api_key=fred_api_key)
            # 1-year treasury (DGS1)
            series_id = 'DGS1'
            rate_data = fred.get_series(series_id)
            if len(rate_data) == 0:
                risk_free_rate = 3.0
            else:
                latest_value = rate_data.dropna().iloc[-1]
                risk_free_rate = float(latest_value)
        except Exception as e:
            print(f"[WARN] Could not fetch rate from FRED: {e}")
            risk_free_rate = 3.0

    return jsonify({
        "price": current_price,
        "risk_free_rate": risk_free_rate,
        "expiriesData": expiries_data
    })


@app.route('/calculate', methods=['POST'])
def calculate():
    """
    1) Get user input from the form.
    2) Re-fetch current price & compute historical vol (1y).
    3) Compute BS + Monte Carlo.
    4) Generate a simple P/L payoff chart at expiration.
    5) Return results + embedded chart.
    """
    ticker_symbol = request.form.get('ticker', '').strip()
    expiration_date_str = request.form.get('expiration_date', '').strip()
    option_type = request.form.get('option_type', 'call').lower()
    strike_price_str = request.form.get('strike_price', '0')
    risk_free_rate_str = request.form.get('risk_free_rate', '0')

    try:
        K = float(strike_price_str)
        r = float(risk_free_rate_str) / 100.0
    except ValueError:
        return "Error: Invalid strike price or risk-free rate."

    # 1) Fetch price from yfinance
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        hist = ticker_data.history(period='1d')
        if hist.empty:
            return f"Error: Could not fetch current price for {ticker_symbol}."
        current_price = float(hist['Close'].iloc[-1])
    except Exception as e:
        return f"Error fetching price from yfinance: {e}"

    # 2) Compute volatility from last year
    try:
        hist_1y = ticker_data.history(period='1y')
        if len(hist_1y) < 2:
            return f"Error: Not enough historical data for vol calculation of {ticker_symbol}."
        hist_1y['Returns'] = hist_1y['Close'].pct_change()
        daily_std = hist_1y['Returns'].std()
        sigma = daily_std * np.sqrt(252)
    except Exception as e:
        return f"Error computing volatility: {e}"

    # 3) DTE
    try:
        expiration_date = datetime.datetime.strptime(expiration_date_str, "%Y-%m-%d")
        today = datetime.datetime.today()
        days_to_expiry = (expiration_date - today).days
        T = max(days_to_expiry, 0) / 365.0
    except:
        return "Error: Invalid expiration date format."

    # 4) Compute BS & Monte Carlo
    if option_type == 'call':
        option_value_bs = black_scholes_call_price(current_price, K, T, r, sigma)
    else:
        option_value_bs = black_scholes_put_price(current_price, K, T, r, sigma)

    option_value_mc = monte_carlo_option_pricing(
        current_price, K, T, r, sigma, option_type
    )

    # -----------------------------------------------
    # 5) Generate P/L payoff chart at maturity
    # -----------------------------------------------

    import io, base64
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lower_bound = 0.7 * current_price
    upper_bound = 1.3 * current_price
    steps = 100
    prices = np.linspace(lower_bound, upper_bound, steps)
    payoff = []

    for p in prices:
        if option_type == 'call':
            payoff_exp = max(p - K, 0)
        else:  # put
            payoff_exp = max(K - p, 0)

        total_pnl = payoff_exp - option_value_bs
        payoff.append(total_pnl)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.plot(prices, payoff, label='P/L at Expiration', color='blue')
    ax.set_xlabel('Underlying Price at Expiration')
    ax.set_ylabel('Profit / Loss (per share)')
    ax.set_title(f'{ticker_symbol.upper()} {option_type.capitalize()} P/L')
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    chart_html = f"""
    <img src="data:image/png;base64,{encoded}"
         alt="P/L Chart"
         style="max-width:100%; border:1px solid #ccc; margin-top:10px;" />
    """

    # -----------------------------------------------
    # 6) Build the HTML with the P/L chart
    # -----------------------------------------------
    result_html = f"""
    <h2>Option Pricing for {ticker_symbol.upper()}</h2>
    <p>Current Price (Yahoo Finance): {current_price:.2f}</p>
    <p>Expiration Date: {expiration_date_str}</p>
    <p>Option Type: {option_type.capitalize()}</p>
    <p>Strike Price: {K:.2f}</p>
    <p>Risk-Free Rate: {r*100:.2f}%</p>
    <p>Annual Volatility (1y hist): {sigma*100:.2f}%</p>
    <p>Time to Expiration: {days_to_expiry} days</p>
    <hr />
    <p><strong>Black–Scholes Price:</strong> ${option_value_bs:.2f}</p>
    <p><strong>Monte Carlo Price:</strong> ${option_value_mc:.2f}</p>
    <hr />
    <p><em>P/L at expiration (assuming 1 long contract, ignoring time decay after purchase):</em></p>
    {chart_html}
    """

    return result_html


if __name__ == '__main__':
    app.run(debug=True)