import numpy as np
from scipy.integrate import quad, cumtrapz,simps

def heston_mc(tipo,S,K,tau,r,q,param,pasosT,iter):

    """
    Precio Heston por Montecarlo
    Inputs
    - tipo: 'C' o 'P'
    - S: Spot
    - K: Strike
    - tau: Tiempo a la madurez en años
    - r: tasa de interes libre de riesgo (anualizada)
    - param: diccionario de paramentros de Heston con la siguiente 
             forma {'rho': - ,'theta': - ,'sigma': - ,'kappa': - ,'v0': -}
            Para el proceso
            # dS_t= r S_t dt + sigma_t S_t dW_t
            # d v_t = kappa*(theta - v_t) dt + sigma sigma_t dW'_t
    - pasosT: Numero de pasos temporales totales
    - iter: Numero de Iteraciones para computar el promedio
    Output
    - precio del call
    
    """
    rho, theta, sigma, kappa, v0 = param.values()
    
    opcion = np.zeros(iter)

    dt = tau/pasosT
    S_t = np.zeros((pasosT, iter))
    V_t = np.zeros((pasosT, iter))
    S_t[0,:] = S
    V_t[0,:] = v0

    cov = np.array([[1,rho],[rho,1]])

    Z = np.random.multivariate_normal([0,0], cov, (iter,pasosT)).T
    Z1=Z[0]
    Z2=Z[1]

    for i in range(1,pasosT):
        vmax = np.maximum(V_t[i-1,:],0)
        V_t[i,:] = V_t[i-1,:] + kappa * (theta - vmax)* dt + sigma *  np.sqrt(vmax * dt) * Z2[i,:]
        S_t[i,:] = S_t[i-1,:] + r*S_t[i-1,:] * dt + np.sqrt(vmax * dt) * S_t[i-1,:] * Z1[i,:] 

    if tipo == 'C':
        payoff = np.maximum(S_t[-1,:] - K,0)
    elif tipo == 'P':
        payoff = np.maximum(K- S_t[-1,:],0)

    opcion = np.exp(-r*tau)*payoff

    return opcion.mean()

def phi_heston(u,S0,t,r,q,param):
    """
    Sacado de Shoutens "A perfect Calibration". Evita la Heston Trap. Funcion caracteristica de Heston
    """
    rho, theta, sigma, kappa, v0 = param.values()
    
    d = np.sqrt((rho*sigma*u*1j - kappa)**2 + (sigma**2)*(1j*u + u**2))
    g = (kappa- rho*sigma*u*1j - d)/(kappa- rho*sigma*u*1j + d)
    
    ex1 = np.exp(1j*u*(np.log(S0) + (r-q)*t))
    ex2 = np.exp(theta*kappa*(1/(sigma**2))*((kappa- rho*sigma*u*1j - d)*t - 2*np.log((1-g*np.exp(-d*t))/(1-g))))
    ex3 = np.exp(v0*(1/(sigma**2))*((kappa- rho*sigma*u*1j - d)*((1-np.exp(-d*t))/(1-g*np.exp(-d*t)))))
    
    return ex1*ex2*ex3

def FTintegrandalpha(u,alpha,S0,K,t,r,q,param):
    """
    Con alpha para que phi sea cuadrado integrable, segun Carr Madan 1999
    """
    denom = alpha**2 + alpha - u**2 + 1j*(2*alpha+1)*u
    num = np.exp(-r*t)*phi_heston((u-(alpha+1)*1j),S0,t,r,q,param)
    return np.real(np.exp(-1j*u*np.log(K))*(num/denom))

def alphaCF_Heston_price(dx,alpha,S0,K,t,r,q,param):
    """
    Precio Call Heston por CF integrando por simpson 
    Inputs
    - dx: step de integracion
    - alpha: parametro regularizador
    - S0: Spot
    - K: Strike
    - t: Tiempo a la madurez en años
    - r: tasa de interes libre de riesgo (anualizada)
    - q: dividendos
    - param: diccionario de paramentros de Heston con la siguiente 
             forma {'rho': - ,'theta': - ,'sigma': - ,'kappa': - ,'v0': -}
            Para el proceso
            dS_t= r S_t dt + sigma_t S_t dW_t
            d v_t = kappa*(theta - v_t) dt + sigma sigma_t dW'_t
    Output
    - precio del call
    
    """
    u=np.arange(0,1000,step=dx)
    y= FTintegrandalpha(u,alpha,S0,K,t,r,q,param)
    Integral = simps(y,x=u)
    return np.exp(-alpha*np.log(K))*Integral / np.pi

def phi_hestonWJ(u,S0,t,r,q,param):
    """
    Sacado de Shoutens y Magnusson. Funcion caracteristica de Heston With Jumps
    """
    rho, theta, sigma, kappa, v0, landa,muJ,sigmaJ = param.values()
    
    d = np.sqrt((rho*sigma*u*1j - kappa)**2 + (sigma**2)*(1j*u + u**2))
    g = (kappa- rho*sigma*u*1j - d)/(kappa- rho*sigma*u*1j + d)
    
    ex1 = np.exp(1j*u*(np.log(S0) + (r-q)*t))
    ex2 = np.exp(theta*kappa*(1/(sigma**2))*((kappa- rho*sigma*u*1j - d)*t - 2*np.log((1-g*np.exp(-d*t))/(1-g))))
    ex3 = np.exp(v0*(1/(sigma**2))*((kappa- rho*sigma*u*1j - d)*((1-np.exp(-d*t))/(1-g*np.exp(-d*t)))))
    ex4 = np.exp(-landa*muJ*1j*u*t + landa*t*(((np.power(1+muJ,1j*u))*np.exp((sigmaJ**2)*(1j*u-1)))-1))
    return ex1*ex2*ex3*ex4

def FTintegrandalphaWJ(u,alpha,S0,K,t,r,q,param):
    """
    Con alpha para que phi sea cuadrado integrable, segun Carr Madan 1999
    """
    denom = alpha**2 + alpha - u**2 + 1j*(2*alpha+1)*u
    num = np.exp(-r*t)*phi_hestonWJ((u-(alpha+1)*1j),S0,t,r,q,param)
    return np.real(np.exp(-1j*u*np.log(K))*(num/denom))

def alphaCF_HestonWJ_price(dx,alpha,S0,K,t,r,q,param):

    """
    Precio Call Heston por CF integrando por simpson 
    Inputs
    - dx: step de integracion
    - alpha: parametro regularizador
    - S0: Spot
    - K: Strike
    - t: Tiempo a la madurez en años
    - r: tasa de interes libre de riesgo (anualizada)
    - q: dividendos
    - param: diccionario de paramentros de Heston con la siguiente 
            forma {'rho': - ,'theta': - ,'sigma': - ,'kappa': - ,'v0': -, 'landa': -, 'muJ': - ,'sigmaJ': -}
            Para el proceso
            dS_t= (r-landa*muJ) S_t dt + sigma_t S_t dW_t + J_t dN_t
            d v_t = kappa*(theta - v_t) dt + sigma sigma_t dZ_t
            dW dZ= rho dt

            con log(1+J_t) \sim N( mu=log(1+muJ) - 0.5*sigmaJ^2, var = sigma_j^2)
    Output
    - precio del call
    
    """
    u=np.arange(0,1000,step=dx)
    y= FTintegrandalphaWJ(u,alpha,S0,K,t,r,q,param)
    Integral = simps(y,x=u)
    return np.exp(-alpha*np.log(K))*Integral / np.pi

