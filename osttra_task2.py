import pandas as pd
import numpy as np
from scipy.stats import norm

def option_price(S, K, T, r, sigma, option_type, q):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "CALL":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == "PUT":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return price, delta, vega

def quanto_option_price(S, K, r_underlying, r_payment, sigma, T, sigma_fx, rho, option_type, q):
    mu_quanto = r_underlying - q - rho * sigma * sigma_fx
    d1 = (np.log(S / K) + (mu_quanto + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "CALL":
        price = (S * np.exp((mu_quanto - r_payment) * T) * norm.cdf(d1) - K * np.exp(-r_payment * T) * norm.cdf(d2))
        delta = np.exp(-mu_quanto * T) * norm.cdf(d1)
    elif option_type == "PUT":
        price = K * np.exp(-r_payment * T) * norm.cdf(-d2) + S * np.exp((mu_quanto - r_payment) * T) * norm.cdf(-d1)
        delta = np.exp(-mu_quanto * T) * (norm.cdf(d1) - 1)
    
    vega = S * np.exp(-mu_quanto * T) * np.sqrt(T) * norm.pdf(d1)
    return price, delta, vega
def intrest_rate(curr):
    if curr == "USD":
        r = 0.05
    elif curr == "EUR":
        r = 0.03
    else:
        r = 0.02
    return r
def compute_pv(row):
    trade_type = row['type']
    quantity = row['quantity']
    underlying = row['underlying']
    S = market_data.loc[market_data["underlying"] == underlying, "spot_price"].values[0]
    sigma = market_data.loc[market_data["underlying"] == underlying, "volatility"].values[0]
    option_type = row['call_put']
    K = row['strike']
    T = row['expiry']
    payment_currency = row['payment_currency']
    
    r=intrest_rate(payment_currency)


    if trade_type == "REGULAR":
        price, D, v = option_price(S, K, T, r, sigma, option_type, q=0)
    elif trade_type == "ODD":
        r_underlying = intrest_rate("USD")
        r_payment = r
        sigma_fx = market_data.loc[market_data["underlying"] == "EUR/USD", "volatility"].values[0]
        price, D, v = quanto_option_price(S, K, r_underlying, r_payment, sigma, T, sigma_fx, 0.5, option_type, 0)
    pv = quantity * price
    D_pv = quantity * D
    v_pv = quantity * v
    return pv, D_pv, v_pv


results=[]
# Load data
market_data = pd.read_csv("C:/Users/Zackarias/OneDrive/osttra/market_data.csv")
trade_data = pd.read_csv("C:/Users/Zackarias/OneDrive/osttra/trade_data.csv")

# Compute PV, delta, and vega for each trade
results = trade_data.apply(compute_pv, axis=1, result_type='expand')
results.columns = ["pv", "equity_delta", "equity_vega"]

# Combine results with trade_data
results["trade_id"] = trade_data["trade_id"]
results = results[["trade_id", "pv", "equity_delta", "equity_vega"]]

# Calculate average equity vega
average_equity_vega = results["equity_vega"].mean()
print("Average Equity Vega:", average_equity_vega)
print(results)







