import numpy as np
from scipy.interpolate import interp2d
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize, brentq
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


# --- Heston Call price ---
def heston_call_price(params, S0, K, T, r):
    def integrand(phi):
        cf1         = vm.heston_char_func(phi - 1j, params, S0, r, T)
        cf2         = vm.heston_char_func(-1j, params, S0, r, T)
        numerator   = np.exp(-1j * phi * np.log(K)) * cf1
        denominator = 1j * phi * cf2
        return (numerator / denominator).real

    integral   = quad(integrand, 0, 100, limit=100)[0]
    call_price = S0 - np.sqrt(S0 * K) / np.pi * integral * np.exp(-r * T)
    return call_price


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


# --- Call price surface optimmized with Heston Model ---
def heston_call_price_surface(params, S0, K_array, T, r, n_integration=128):
    # grille de Fourier (phi)
    phi = np.linspace(1e-6, 200, n_integration)
    cf1 = vm.heston_char_func(phi - 1j, params, S0, r, T)
    cf2 = vm.heston_char_func(-1j, params, S0, r, T)

    logK        = np.log(K_array)
    numerator   = np.exp(-1j * np.outer(phi, logK)) * cf1[:, None]
    denominator = (1j * phi)[:, None] * cf2[:, None]
    integrand   = (numerator / denominator).real
    integral    = np.trapz(integrand, phi, axis=0)

    return S0 - np.sqrt(S0 * K_array) / np.pi * integral * np.exp(-r * T)


if __name__=='__main__':

    # Paramètres
    S0    = 100
    r     = 0.05
    K     = 95
    T     = 0.5
    sigma = 0.2

    strikes    = np.linspace(80, 120, 10)
    maturities = np.linspace(0.1, 2.0, 5)

    call_bs = bs_call_price(S0, K, T, r, sigma)
    put_bs  = bs_put_price(S0, K, T, r, sigma)

    vol_surface   = vm.synth_implied_vol_surface(S0, strikes, maturities)
    price_surface = call_price_surface(S0, r, strikes, maturities, vol_surface)
    

    # --- Heston Volatility Surface ---
    params_init = [2.0, 0.04, 0.3, -0.5, 0.04]  # kappa, theta, sigma, rho, v0
    bounds      = [(0.01, 10), (0.001, 1), (0.01, 1), (-0.99, 0.99), (0.001, 1)]

    # Calibration 
    result = minimize(vm.calibration_error, params_init,
                  args=(S0, r, strikes, maturities, vol_surface),
                  bounds=bounds, method='L-BFGS-B',
                  options={'disp': True, 'maxiter': 100})

    # --- Construction des vols du modèle calibré ---
    model_vols = np.zeros_like(vol_surface)
    for i, T in enumerate(maturities):
        heston_prices = heston_call_price_surface(result.x, S0, strikes, T, r)
    
    print(heston_prices)

