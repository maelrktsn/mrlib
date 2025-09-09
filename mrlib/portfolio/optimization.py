import pandas as pd
import numpy as np
from itertools import repeat
from scipy.optimize import minimize

# --- Equal Risk Contribution Optimization ---
def opti_erc(price:pd.DataFrame):

    def calculate_portfolio_var(w,V):
        # function that calculates portfolio risk
        w = np.matrix(w)
        return (w*V*w.T) [0,0]

    def calculate_risk_contribution(w,V):
        # function that calculates asset contribution to total risk
        w     = np.matrix(w)
        sigma = np.sqrt(calculate_portfolio_var(w,V))
        MRC   = V*w.T # Marginal Risk Contribution
        RC    = np.multiply(MRC,w.T)/sigma # Risk Contribution
        return RC

    def risk_budget_objective (x,pars):
        # calculate portfolio risk
        V           = pars[0]# covariance table
        x_t         = pars[1] # risk target in percent of portfolio risk
        sig_p       = np.sqrt(calculate_portfolio_var(x, V)) # portfolio sigma
        risk_target = np.asmatrix(np.multiply(sig_p,x_t))
        asset_RC    = calculate_risk_contribution(x,V)
        J           = sum(np.square (asset_RC-risk_target.T))[0,0] # sum of squared error
        return J

    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    
    ret     = price.pct_change().iloc[1:,]
    mat_vcv = np.matrix(ret.cov())
    x_t     = np.repeat(1/(len(mat_vcv)), len(mat_vcv)).tolist() # risk budget of total percent of total portfolio risk (equal risk)
    w0      = x_t
    cons    = ({'type': 'eq', 'fun': total_weight_constraint}, {'type': 'ineq', 'fun': long_only_constraint})
    res     = minimize(risk_budget_objective, w0, args=[mat_vcv, x_t], method = 'SLSQP', constraints=cons, options={'disp': True, 'ftol' : 1e-17, 'maxiter':1000})
    w_rb    = np.asmatrix(res.x)

    erc_weight = pd.DataFrame(w_rb, columns = price.columns)

    return erc_weight

# --- Max Sharpe Ratio Optimization ---
def opti_max_sharpe(price:pd.DataFrame):

    def get_ret_vol_sr(weights):
        weights = np.array(weights)
        ret     = np.sum(df_ret.mean()* weights)*252
        vol     = np.sqrt(np.dot(weights.T, np.dot(df_ret.cov()*252, weights)))
        sr      = ret/vol    
        return np.array([ret, vol, sr])
    
    def neg_sharpe_ratio(weights):
        return get_ret_vol_sr(weights)[2] * -1


    def check_sum(weights):
        # return 0 if sum is equal to 1
        return np.sum(weights)-1

    df_ret = price.pct_change().iloc[1:,]

    # Optimization Constraints 
    cons        = ({'type':'eq', 'fun':check_sum})
    bound       = (0,1)
    bounds      = tuple(repeat(bound, len(price.columns)))
    init_weight = [1/len(price.columns)]
    init_guess  = init_weight * len(price.columns)

    opt_results        = minimize(neg_sharpe_ratio, init_guess, method ='SLSQP', bounds = bounds, constraints = cons)
    max_sharpe_weight  = pd.DataFrame(opt_results.x, columns = price.columns)

    return max_sharpe_weight


