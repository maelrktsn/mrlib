import numpy as np 
from mrlib.pricing import option as opt
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import matplotlib.pyplot as plt

# --- Synthetic Implied Volality Surface Example ---
def synth_implied_vol_surface(S0, strikes, maturities):
    vol_surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
            vol_surface[i, :] = 0.2 + 0.1 * np.exp(-T) + 0.15 * ((strikes / S0 - 1) ** 2) # Exemple : volatilité implicite décroissante en maturité et smile en strike
    return vol_surface


# --- Implied volatility via inversion BS ---
def implied_volatility(price, S, K, T, r, sigma_init=0.2):
    def objective(sigma):
        return opt.bs_call_price(S, K, T, r, sigma) - price
    try:
        vol = brentq(objective, 1e-6, 5.0)
    except:
        vol = np.nan
    return vol
    

# --- Local Volatility Calculation (Dupire Model) ---
def dupire_local_vol(S0, r, strikes, maturities, price_surface):
    dK        = strikes[1] - strikes[0]
    dT        = maturities[1] - maturities[0]
    local_vol = np.zeros_like(price_surface)

    # Interpolations pour dérivées (central differences)
    for i in range(1, len(maturities) - 1):
        for j in range(1, len(strikes) - 1):
            C         = price_surface[i, j]
            C_T_plus  = price_surface[i + 1, j]
            C_T_minus = price_surface[i - 1, j]
            C_K_plus  = price_surface[i, j + 1]
            C_K_minus = price_surface[i, j - 1]

            C_KK = (C_K_plus - 2 * C + C_K_minus) / (dK ** 2)
            C_T  = (C_T_plus - C_T_minus) / (2 * dT)
            C_K  = (C_K_plus - C_K_minus) / (2 * dK)

            K           = strikes[j]
            numerator   = C_T + r * K * C_K
            denominator = 0.5 * (K ** 2) * C_KK

            if denominator > 0 and numerator > 0:
                local_vol[i, j] = np.sqrt(numerator / denominator)
            else:
                local_vol[i, j] = np.nan  # valeurs non définies

    return local_vol


# --- Heston characteristic function---
def heston_char_func(u, params, S0, r, T):
    kappa, theta, sigma, rho, v0 = params

    x = np.log(S0)
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * 1j * u - b)**2 + (sigma**2) * (1j * u + u**2))
    g = (b - rho * sigma * 1j * u - d) / (b - rho * sigma * 1j * u + d)

    exp1 = 1j * u * (x + r * T)
    exp2 = (a / sigma**2) * ((b - rho * sigma * 1j * u - d) * T
                             - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    exp3 = (v0 / sigma**2) * (b - rho * sigma * 1j * u - d) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))

    return np.exp(exp1 + exp2 - exp3)


# --- Heston error function for calibration ---
def calibration_error(params, S0, r, strikes, maturities, market_vols):
    error = 0.0
    count = 0.0
    for i, T in enumerate(maturities):
            prices = opt.heston_call_price_surface(params, S0, strikes, T, r)
            vols   = np.array([implied_volatility(p, S0, k, T, r) for p, k in zip(prices, strikes)])
            mask   = ~np.isnan(vols)
            if np.any(mask):
                error += np.sum((vols[mask] - market_vols[i, mask])**2)
                count += np.sum(mask)
    return error / count if count > 0 else np.inf


if __name__=='__main__':

    # Paramètres
    S0 = 100
    r  = 0.05

    strikes    = np.linspace(80, 120, 10)
    maturities = np.linspace(0.1, 2.0, 5)

    # --- Dupire Volatility Surface ---
    market_vol_surface = synth_implied_vol_surface(S0, strikes, maturities)
    price_surface      = opt.call_price_surface(S0, r, strikes, maturities, market_vol_surface)
    local_vol_surface  = dupire_local_vol(S0, r, strikes, maturities, price_surface)

    # Plot
    K_grid, T_grid = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K_grid, T_grid, local_vol_surface, cmap='viridis')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Local Volatility')
    ax.set_title('Surface de volatilité locale calibrée (Dupire)')
    plt.show()

    # --- Heston Volatility Surface ---
    params_init = [2.0, 0.04, 0.3, -0.5, 0.04]  # kappa, theta, sigma, rho, v0
    bounds      = [(0.01, 10), (0.001, 1), (0.01, 1), (-0.99, 0.99), (0.001, 1)]

    # Calibration 
    result = minimize(calibration_error, params_init,
                  args=(S0, r, strikes, maturities, market_vol_surface),
                  bounds=bounds, method='L-BFGS-B',
                  options={'disp': True, 'maxiter': 100})
    
    print("\nParamètres calibrés :")
    print(f"kappa = {result.x[0]:.4f}")
    print(f"theta = {result.x[1]:.4f}")
    print(f"sigma = {result.x[2]:.4f}")
    print(f"rho   = {result.x[3]:.4f}")
    print(f"v0    = {result.x[4]:.4f}")
    print(f"Erreur calibrée = {result.fun:.6f}")


    # --- Construction des vols du modèle calibré ---
    model_vols = np.zeros_like(market_vol_surface)
    for i, T in enumerate(maturities):
        prices           = opt.heston_call_price_surface(result.x, S0, strikes, T, r)
        model_vols[i, :] = np.array([implied_volatility(p, S0, k, T, r) for p, k in zip(prices, strikes)])

    # --- Plot ---
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    fig            = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(K_grid, T_grid, market_vol_surface, cmap='viridis')
    ax1.set_title('Volatilité implicite marché')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Maturité')
    ax1.set_zlabel('Volatilité')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(K_grid, T_grid, model_vols, cmap='viridis')
    ax2.set_title('Volatilité implicite modèle calibré')
    ax2.set_xlabel('Strike')
    ax2.set_ylabel('Maturité')
    ax2.set_zlabel('Volatilité')

    plt.show()