import pandas as pd
import numpy as np
from scipy.stats import norm
import argparse
# REGULAR option
def option_price(S, K, T, r, sigma, option_type, q, Tp):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "CALL":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == "PUT":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    
    vega = S * norm.pdf(d1) * np.sqrt(T)
    # discount due to delay of payment
    price = price * np.exp(-r*(Tp-T))
    return price, delta, vega
# ODD option
# r_underlying/r_foreign
# r_payment/r_domestic
# sigma underlying volatility
# sigma_fx exhange rate volatility
# rho correlation equity spot price and an FX spot rate
def quanto_option_price(S, K, r, sigma, T, sigma_fx, rho, option_type, q, Tp, E):
    sigma_quanto = np.sqrt(sigma**2 + sigma_fx**2 - 2 * rho * sigma * sigma_fx)
    d1 = (np.log(S / ((1 / E) * K)) + (r - q + 0.5 * sigma_quanto**2) * T) / (sigma_quanto * np.sqrt(T))
    d2 = d1 - sigma_quanto * np.sqrt(T)
    
    if option_type == "CALL":
        price = (S * np.exp(-q * T) * norm.cdf(d1) - (1 / E) * K * np.exp(-r * T) * norm.cdf(d2))
        delta = np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "PUT":
        price = (1 / E) * K * np.exp(-r * T) * norm.cdf(-d2) + S * np.exp(q * T) * norm.cdf(-d1)
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    price = price * np.exp(-r * (Tp - T))
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
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
    Tp = row['payment_time']
    payment_currency = row['payment_currency']
    
    r=intrest_rate(payment_currency)


    if trade_type == "REGULAR":
        price, D, v = option_price(S, K, T, r, sigma, option_type, 0, Tp)
    elif trade_type == "ODD":
        exchange = str(payment_currency+"/"+"USD")
        E = market_data.loc[market_data["underlying"] == exchange, "spot_price"].values[0]
        sigma_fx = market_data.loc[market_data["underlying"] == exchange, "volatility"].values[0]
        price, D, v = quanto_option_price(S, K, r, sigma, T, sigma_fx, -0.5, option_type, 0, Tp, E)
    # quantities for present value
    pv = quantity * price
    D_pv = quantity * D
    v_pv = quantity * v
    return pv, D_pv, v_pv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate option prices.')
    parser.add_argument('--market_data', type=str, required=True, help='Path to market data CSV file')
    parser.add_argument('--trade_data', type=str, required=True, help='Path to trade data CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    args = parser.parse_args()
    
    market_data = pd.read_csv(args.market_data)
    trade_data = pd.read_csv(args.trade_data)

    # compute PV, delta, and vega 
    results = trade_data.apply(compute_pv, axis=1, result_type='expand')
    results.columns = ["pv", "equity_delta", "equity_vega"]

    results["trade_id"] = trade_data["trade_id"]
    results = results[["trade_id", "pv", "equity_delta", "equity_vega"]]

    results.to_csv(args.output, index=False)