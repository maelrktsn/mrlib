import numpy as np
from scipy.interpolate import interp2d
from scipy.stats import norm
from mrlib.pricing import volatility_models as vm 

# --- European Call Black-Scholes price ---
def bs_call_price(S, K, T, r, sigma):
    if T == 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# --- European Put Black-Scholes price ---
def bs_put_price(S, K, T, r, sigma):
    if T == 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * K * norm.cdf(-d2) - S * norm.cdf(-d1)


# --- Call price surface from Implied Volatility ---
def call_price_surface(S0, r, strikes, maturities, vol_surface):
    price_surface = np.zeros_like(vol_surface)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            sigma               = vol_surface[i, j]
            price_surface[i, j] = bs_call_price(S0, K, T, r, sigma)
    return price_surface


# --- Put price surface from Implied Volatility ---
def put_price_surface(S0, r, strikes, maturities, vol_surface):
    price_surface = np.zeros_like(vol_surface)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            sigma               = vol_surface[i, j]
            price_surface[i, j] = bs_put_price(S0, K, T, r, sigma)
    return price_surface


if __name__=='__main__':

    # Paramètres
    S0    = 100
    r     = 0.05
    K     = 95
    T     = 0.5
    sigma = 0.2

    strikes    = np.linspace(80, 120, 21)
    maturities = np.linspace(0.1, 2.0, 21)

    call_bs = bs_call_price(S0, K, T, r, sigma)
    put_bs  = bs_put_price(S0, K, T, r, sigma)

    vol_surface   = vm.synth_implied_vol_surface(S0, strikes, maturities)
    price_surface = call_price_surface(S0, r, strikes, maturities, vol_surface)

