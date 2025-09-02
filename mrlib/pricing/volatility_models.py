import numpy as np 
from mrlib.pricing import option as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def synth_implied_vol_surface(S0, strikes, maturities):
    """
    Exemple de surface synthétique de volatilité implicite.
    """
    vol_surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            # Exemple : volatilité implicite décroissante en maturité et smile en strike
            vol_surface[i, j] = 0.2 + 0.1 * np.exp(-T) + 0.15 * ((K / S0 - 1) ** 2)
    return vol_surface


def dupire_local_vol(S0, r, strikes, maturities, price_surface):
    """
    Calcule la volatilité locale avec la formule de Dupire,
    à partir de la surface de prix d’options (prix calls)
    """
    dK = strikes[1] - strikes[0]
    dT = maturities[1] - maturities[0]
    local_vol = np.zeros_like(price_surface)

    # Interpolations pour dérivées (central differences)
    for i in range(1, len(maturities) - 1):
        for j in range(1, len(strikes) - 1):
            C = price_surface[i, j]
            C_T_plus = price_surface[i + 1, j]
            C_T_minus = price_surface[i - 1, j]
            C_K_plus = price_surface[i, j + 1]
            C_K_minus = price_surface[i, j - 1]

            C_KK = (C_K_plus - 2 * C + C_K_minus) / (dK ** 2)
            C_T = (C_T_plus - C_T_minus) / (2 * dT)
            C_K = (C_K_plus - C_K_minus) / (2 * dK)

            K = strikes[j]
            numerator = C_T + r * K * C_K
            denominator = 0.5 * (K ** 2) * C_KK

            if denominator > 0 and numerator > 0:
                local_vol[i, j] = np.sqrt(numerator / denominator)
            else:
                local_vol[i, j] = np.nan  # valeurs non définies

    return local_vol



if __name__=='__main__':

    # Paramètres
    S0 = 100
    r  = 0.05

    strikes    = np.linspace(80, 120, 21)
    maturities = np.linspace(0.1, 2.0, 21)

    vol_surface       = synth_implied_vol_surface(S0, strikes, maturities)
    price_surface     = opt.call_price_surface(S0, r, strikes, maturities, vol_surface)
    local_vol_surface = dupire_local_vol(S0, r, strikes, maturities, price_surface)

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