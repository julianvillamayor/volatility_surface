import math
from scipy.stats import norm
from scipy.optimize import fsolve, brentq


def opcion_europea_bs(tipo, S, K, T, r, sigma, div):
    """
    opcion_europea_bs
    Def
    Calculador del precio de una opcion Europea con el modelo de Black Scholes
    Inputs
    - tipo : string - Tipo de contrato entre ["CALL","PUT"]
    - S : float - Spot price del activo
    - K : float - Strike price del contrato
    - T : float - Tiempo hasta la expiracion (en años)
    - r : float - Tasa 'libre de riesgo' (anualizada)
    - sigma : float - Volatilidad implicita (anualizada)
    - div : float - Tasa de dividendos continuos (anualizada)
    Outputs
    - precio_BS: float - Precio del contrato
    """
    #Defino los ds
    d1 = (math.log(S / K) + (r - div + 0.5 * sigma * sigma) * T) / sigma / math.sqrt(T)
    d2 = (math.log(S / K) + (r - div - 0.5 * sigma * sigma) * T) / sigma / math.sqrt(T)

    if (tipo == "C"):
        precio_BS = math.exp(-div*T) *S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif (tipo == "P"):
        precio_BS = K * math.exp(-r * T) * norm.cdf(-d2) - math.exp(-div*T) * S * norm.cdf(-d1)
    return precio_BS

def samesign(a, b):
    return a * b > 0

def bisect(func, low, high, iters=100):
    'Find root of continuous function where f(low) and f(high) have opposite signs'

    assert not samesign(func(low), func(high))

    for i in range(iters):
        midpoint = (low + high) / 2.0
        if samesign(func(low), func(midpoint)):
            low = midpoint
        else:
            high = midpoint

    return midpoint

def impvolfunc_bs(tipo, S, K, T, r, precio_mercado, div):
    """
    impvolfunc_bs 
    input
    - tipo : string - Tipo de contrato entre ["C","P"]
    - S : float - Spot price del activo
    - K : float - Strike price del contrato
    - T : float - Tiempo hasta la expiracion (en años)
    - r : float - Tasa 'libre de riesgo' (anualizada)
    - precio_mercado : float - precio a ajustar con la volatilidad
    - div : float - Tasa de dividendos continuos (anualizada)
    Outputs
    - impvol: float - volatilidad implicita (anualizada)
    """
    func = lambda sigma: (opcion_europea_bs(tipo, S, K, T, r, sigma, div) - precio_mercado)

    impvol = bisect(func,0.00001, 3.0, 100)
    return impvol


def implied_volatility( price, S0, K, T, r,div, tipo="C", method="fsolve", disp=True ):
    """ Devuelve la volatilidad implicita, resolviendo con metodos de scipy que son rapidos.
        methods:  fsolve (default) or brent
    """

    def obj_fun(vol):
        return ( price - opcion_europea_bs(tipo, S0, K, T, r, vol, div) )
    
    if method == "brent":
        x, r = brentq( obj_fun, a=1e-15, b=500, full_output=True)
        if r.converged == True:
            return x
    if method == "fsolve":
        X0 = [0.1, 0.5, 1, 3, 0.02]   # set of initial guess points
        for x0 in X0:
            x, _, solved, _ = fsolve( obj_fun, x0, full_output=True, xtol=1e-8)
            if solved == 1:
                return x[0]  
    
    if disp == True:
        #print("Strike", K)
        return -1 #En caso de que no funcione lo manda a menos uno