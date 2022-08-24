import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from opcion_europea_bs import opcion_europea_bs

def rawSVI(k,a,b,rho,m,sigma):
    return a+b*(rho*(k-m)+np.sqrt((k-m)**2+sigma**2)) # Varianza total parametrizacion cruda

def naturalSVI(k,delta,mu,rho,omega,zeta):
    return delta + 0.5*omega*(1 + zeta*rho*(k-mu) + np.sqrt((zeta*(k-mu) + rho)**2 + (1-rho**2)))

###############

#Estos transforman un set de parametros a otros

def nat2raw(delta,mu,rho,omega,zeta):
    a= delta + 0.5*omega*(1-rho**2)
    b= omega*zeta*0.5
    m= mu - rho/zeta
    sigma= np.sqrt(1-rho**2)/zeta
    return [a,b,rho,m,sigma]

def raw2nat(a,b,rho,m,sigma):
    omega= ((2*b*sigma)/np.sqrt(1-rho**2))
    delta = a - 0.5*omega*(1-rho**2)
    mu = m + ((rho*sigma)/np.sqrt(1-rho**2))
    zeta= np.sqrt(1-rho**2)/sigma
    return [delta,mu,rho,omega,zeta]

def raw2WJ(t,a,b,rho,m,sigma):
    vt = (a+b*(-rho*m + np.sqrt(m**2 + sigma**2)))/t
    wt = t*vt
    psit=(0.5*b/np.sqrt(wt))*(-(m/(np.sqrt(m**2 + sigma**2))) +rho)
    pt=(1/np.sqrt(wt))*b*(1-rho)
    ct=(1/np.sqrt(wt))*b*(1+rho)
    v_t=(1/t)*(a+b*sigma*np.sqrt(1-rho**2))
    return [vt,psit,pt,ct,v_t]

def WJ2raw(t,vt,wt,psit,pt,ct,v_t):
    wt = t*vt
    
    b = 0.5*np.sqrt(wt)*(ct+pt)
    rho= 1 - ((pt*np.sqrt(wt))/b)
    
    beta = rho - ((2*psit*np.sqrt(wt))/b) 
    alfa = np.sign(beta)*np.sqrt((1/beta**2) - 1)

    m= ((vt-v_t)*t)/(b*(-rho + np.sign(alfa)*np.sqrt(1+ alfa**2) - alfa*np.sqrt(1-rho**2)))
    sigma = alfa*m
    a=v_t*t - b*sigma*np.sqrt(1-rho**2)

    return [a,b,rho,m,sigma]

################

def calendar_arbitrage(setT1,setT2):
    
  a1,b1,r1,m1,s1 = setT1
  a2,b2,r2,m2,s2 = setT2

  #Coeficientes Para resolver una cuartica, que salen de igualar los rawSVI de T1 y T2
  q2 = (1000000 * -2 * (-3 * b1 ** 4 * m1 ** 2 + b1 ** 2 * b2 ** 2 * m1 ** 2 + 4 * b1 ** 2 * b2 ** 2 * m1 * m2 + 
            b1 ** 2 * b2 ** 2 * m2 ** 2 - 3 * b2 ** 4 * m2 ** 2 + 6 * b1 ** 4 * m1 ** 2 * r1 ** 2 + 
            b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 + 4 * b1 ** 2 * b2 ** 2 * m1 * m2 * r1 ** 2 + 
            b1 ** 2 * b2 ** 2 * m2 ** 2 * r1 ** 2 - 3 * b1 ** 4 * m1 ** 2 * r1 ** 4 - 6 * b1 ** 3 * b2 * m1 ** 2 * r1 * r2 - 
            6 * b1 ** 3 * b2 * m1 * m2 * r1 * r2 - 6 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 - 
            6 * b1 * b2 ** 3 * m2 ** 2 * r1 * r2 + 6 * b1 ** 3 * b2 * m1 ** 2 * r1 ** 3 * r2 + 
            6 * b1 ** 3 * b2 * m1 * m2 * r1 ** 3 * r2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * r2 ** 2 + 
            4 * b1 ** 2 * b2 ** 2 * m1 * m2 * r2 ** 2 + b1 ** 2 * b2 ** 2 * m2 ** 2 * r2 ** 2 + 6 * b2 ** 4 * m2 ** 2 * r2 ** 2 - 
            3 * b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 * r2 ** 2 - 12 * b1 ** 2 * b2 ** 2 * m1 * m2 * r1 ** 2 * r2 ** 2 - 
            3 * b1 ** 2 * b2 ** 2 * m2 ** 2 * r1 ** 2 * r2 ** 2 + 6 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 ** 3 + 
            6 * b1 * b2 ** 3 * m2 ** 2 * r1 * r2 ** 3 - 3 * b2 ** 4 * m2 ** 2 * r2 ** 4 - 
            a1 ** 2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2)) - 
            a2 ** 2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2)) - 
            2 * a2 * (3 * b1 ** 3 * m1 * r1 * (-1 + r1 ** 2) - b1 ** 2 * b2 * (2 * m1 + m2) * (-1 + 
                3 * r1 ** 2) * r2 - 3 * b2 ** 3 * m2 * r2 * (-1 + r2 ** 2) + b1 * b2 ** 2 * (m1 + 2 * m2) * 
               r1 * (-1 + 3 * r2 ** 2)) + 2 * a1 * (3 * b1 ** 3 * m1 * r1 * (-1 + r1 ** 2) - 
              b1 ** 2 * b2 * (2 * m1 + m2) * (-1 + 3 * r1 ** 2) * r2 - 3 * b2 ** 3 * m2 * r2 * (-1 + 
                r2 ** 2) + b1 * b2 ** 2 * (m1 + 2 * m2) * r1 * (-1 + 3 * r2 ** 2) + 
              a2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2))) - 
            b1 ** 4 * s1 ** 2 + b1 ** 2 * b2 ** 2 * s1 ** 2 + b1 ** 4 * r1 ** 2 * s1 ** 2 - 2 * b1 ** 3 * b2 * r1 * r2 * 
             s1 ** 2 + b1 ** 2 * b2 ** 2 * r2 ** 2 * s1 ** 2 + b1 ** 2 * b2 ** 2 * s2 ** 2 - b2 ** 4 * s2 ** 2 + 
            b1 ** 2 * b2 ** 2 * r1 ** 2 * s2 ** 2 - 2 * b1 * b2 ** 3 * r1 * r2 * s2 ** 2 + b2 ** 4 * r2 ** 2 * s2 ** 2))

  q4 = (1000000 * (b1 ** 4 * (-1 + r1 ** 2) ** 2 - 4 * b1 ** 3 * b2 * r1 * (-1 + r1 ** 2) * r2 - 
            4 * b1 * b2 ** 3 * r1 * r2 * (-1 + r2 ** 2) + b2 ** 4 * (-1 + r2 ** 2) ** 2 + 
            2 * b1 ** 2 * b2 ** 2 * (-1 - r2 ** 2 + r1 ** 2 * (-1 + 3 * r2 ** 2))))

  q0 = (1000000 * (a1 ** 4 + a2 ** 4 + b1 ** 4 * m1 ** 4 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 + b2 ** 4 * m2 ** 4 - 
            2 * b1 ** 4 * m1 ** 4 * r1 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r1 ** 2 + b1 ** 4 * m1 ** 4 * r1 ** 4 + 
            4 * b1 ** 3 * b2 * m1 ** 3 * m2 * r1 * r2 + 4 * b1 * b2 ** 3 * m1 * m2 ** 3 * r1 * r2 - 
            4 * b1 ** 3 * b2 * m1 ** 3 * m2 * r1 ** 3 * r2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r2 ** 2 - 
            2 * b2 ** 4 * m2 ** 4 * r2 ** 2 + 6 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r1 ** 2 * r2 ** 2 - 
            4 * b1 * b2 ** 3 * m1 * m2 ** 3 * r1 * r2 ** 3 + b2 ** 4 * m2 ** 4 * r2 ** 4 + 
            4 * a2 ** 3 * (b1 * m1 * r1 - b2 * m2 * r2) - 4 * a1 ** 3 * (a2 + b1 * m1 * r1 - 
              b2 * m2 * r2) + 2 * b1 ** 4 * m1 ** 2 * s1 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m2 ** 2 * s1 ** 2 - 
            2 * b1 ** 4 * m1 ** 2 * r1 ** 2 * s1 ** 2 + 4 * b1 ** 3 * b2 * m1 * m2 * r1 * r2 * s1 ** 2 - 
            2 * b1 ** 2 * b2 ** 2 * m2 ** 2 * r2 ** 2 * s1 ** 2 + b1 ** 4 * s1 ** 4 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * s2 ** 2 + 
            2 * b2 ** 4 * m2 ** 2 * s2 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 * s2 ** 2 + 
            4 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 * s2 ** 2 - 2 * b2 ** 4 * m2 ** 2 * r2 ** 2 * s2 ** 2 - 
            2 * b1 ** 2 * b2 ** 2 * s1 ** 2 * s2 ** 2 + b2 ** 4 * s2 ** 4 + 4 * a2 * (b1 * m1 * r1 - b2 * m2 * r2) * 
             (-2 * b1 * b2 * m1 * m2 * r1 * r2 + b1 ** 2 * (m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) + 
              b2 ** 2 * (m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2)) - 4 * a1 * (a2 + b1 * m1 * r1 - 
              b2 * m2 * r2) * (a2 ** 2 - 2 * b1 * b2 * m1 * m2 * r1 * r2 + 2 * a2 * (b1 * m1 * r1 - 
                b2 * m2 * r2) + b1 ** 2 * (m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) + 
              b2 ** 2 * (m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2)) + 2 * a2 ** 2 * 
             (-6 * b1 * b2 * m1 * m2 * r1 * r2 + b1 ** 2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) - s1 ** 2) + 
              b2 ** 2 * (m2 ** 2 * (-1 + 3 * r2 ** 2) - s2 ** 2)) + 
            2 * a1 ** 2 * (3 * a2 ** 2 - 6 * b1 * b2 * m1 * m2 * r1 * r2 + 6 * a2 * (b1 * m1 * r1 - 
                b2 * m2 * r2) + b1 ** 2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) - s1 ** 2) + 
              b2 ** 2 * (m2 ** 2 * (-1 + 3 * r2 ** 2) - s2 ** 2))))


  q3 = (1000000 * -4 * (b1 ** 4 * m1 * (-1 + r1 ** 2) ** 2 - b1 ** 3 * r1 * (-1 + r1 ** 2) * 
             (a1 - a2 + b2 * (3 * m1 + m2) * r2) + b2 ** 3 * (-1 + r2 ** 2) * 
             ((a1 - a2) * r2 + b2 * m2 * (-1 + r2 ** 2)) + b1 * b2 ** 2 * r1 * 
             (a1 - 3 * a1 * r2 ** 2 - b2 * (m1 + 3 * m2) * r2 * (-1 + r2 ** 2) + 
              a2 * (-1 + 3 * r2 ** 2)) + b1 ** 2 * b2 * ((a1 - a2) * (-1 + 3 * r1 ** 2) * r2 + 
              b2 * (m1 + m2) * (-1 - r2 ** 2 + r1 ** 2 * (-1 + 3 * r2 ** 2)))))

  q1 = (1000000 * 4 * (-(b1 ** 4 * m1 ** 3) + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 + b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 - 
            b2 ** 4 * m2 ** 3 + 2 * b1 ** 4 * m1 ** 3 * r1 ** 2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r1 ** 2 + 
            b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r1 ** 2 - b1 ** 4 * m1 ** 3 * r1 ** 4 - b1 ** 3 * b2 * m1 ** 3 * r1 * r2 - 
            3 * b1 ** 3 * b2 * m1 ** 2 * m2 * r1 * r2 - 3 * b1 * b2 ** 3 * m1 * m2 ** 2 * r1 * r2 - b1 * b2 ** 3 * m2 ** 3 * r1 * r2 + b1 ** 3 * b2 * m1 ** 3 * r1 ** 3 * r2 + 3 * b1 ** 3 * b2 * m1 ** 2 * m2 * 
             r1 ** 3 * r2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r2 ** 2 + b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r2 ** 2 + 
            2 * b2 ** 4 * m2 ** 3 * r2 ** 2 - 3 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r1 ** 2 * r2 ** 2 - 
            3 * b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r1 ** 2 * r2 ** 2 + 3 * b1 * b2 ** 3 * m1 * m2 ** 2 * r1 * r2 ** 3 + 
            b1 * b2 ** 3 * m2 ** 3 * r1 * r2 ** 3 - b2 ** 4 * m2 ** 3 * r2 ** 4 + a1 ** 3 * (b1 * r1 - b2 * r2) + 
            a2 ** 3 * (-(b1 * r1) + b2 * r2) + a2 ** 2 * (b1 ** 2 * (m1 - 3 * m1 * r1 ** 2) + 3 * b1 * b2 * (m1 + m2) * r1 * r2 + b2 ** 2 * m2 * (1 - 3 * r2 ** 2)) + 
            a1 ** 2 * (b1 ** 2 * (m1 - 3 * m1 * r1 ** 2) + 3 * b1 * r1 * (-a2 + b2 * (m1 + m2) * r2) + 
              b2 * (3 * a2 * r2 + b2 * (m2 - 3 * m2 * r2 ** 2))) - b1 ** 4 * m1 * s1 ** 2 + 
            b1 ** 2 * b2 ** 2 * m2 * s1 ** 2 + b1 ** 4 * m1 * r1 ** 2 * s1 ** 2 - b1 ** 3 * b2 * m1 * r1 * r2 * s1 ** 2 - 
            b1 ** 3 * b2 * m2 * r1 * r2 * s1 ** 2 + b1 ** 2 * b2 ** 2 * m2 * r2 ** 2 * s1 ** 2 + 
            b1 ** 2 * b2 ** 2 * m1 * s2 ** 2 - b2 ** 4 * m2 * s2 ** 2 + b1 ** 2 * b2 ** 2 * m1 * r1 ** 2 * s2 ** 2 - 
            b1 * b2 ** 3 * m1 * r1 * r2 * s2 ** 2 - b1 * b2 ** 3 * m2 * r1 * r2 * s2 ** 2 + 
            b2 ** 4 * m2 * r2 ** 2 * s2 ** 2 + a2 * (b1 ** 2 * b2 * r2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) + 
                2 * m1 * m2 * (-1 + 3 * r1 ** 2) - s1 ** 2) + b1 ** 3 * r1 * (-3 * m1 ** 2 * 
                 (-1 + r1 ** 2) + s1 ** 2) + b2 ** 3 * r2 * (3 * m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2) + 
              b1 * b2 ** 2 * r1 * (m1 * m2 * (2 - 6 * r2 ** 2) + m2 ** 2 * (1 - 3 * r2 ** 2) + s2 ** 2)) + 
            a1 * (3 * a2 ** 2 * (b1 * r1 - b2 * r2) + a2 * (2 * b1 ** 2 * m1 * (-1 + 3 * r1 ** 2) - 
                6 * b1 * b2 * (m1 + m2) * r1 * r2 + 2 * b2 ** 2 * m2 * (-1 + 3 * r2 ** 2)) + 
              b1 ** 3 * r1 * (3 * m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) + b1 ** 2 * b2 * r2 * ( 
                m1 * m2 * (2 - 6 * r1 ** 2) + m1 ** 2 * (1 - 3 * r1 ** 2) + s1 ** 2) + 
              b1 * b2 ** 2 * r1 * (2 * m1 * m2 * (-1 + 3 * r2 ** 2) + m2 ** 2 * (-1 + 3 * r2 ** 2) - 
                s2 ** 2) + b2 ** 3 * r2 * (-3 * m2 ** 2 * (-1 + r2 ** 2) + s2 ** 2))))

  coef = [q4,q3,q2,q1,q0]
  roots = np.roots(coef) # Para los coeficientes anteriores encuentra todas las raices, imaginarias y reales

  rr = roots[np.isreal(roots)]

  raices = []

  for r in rr: #Quiero comprobar que las raices lo sean efectivamente
    if np.abs(rawSVI(r,a1,b1,r1,m1,s1) - rawSVI(r,a2,b2,r2,m2,s2))<1e-6:
      raices.append(r)
    
  raices = np.sort(np.array(raices))
  num_r = len(raices)

  c = 0 #crossedness
  #Se computa la crossedness como indica Gatheral  
  if num_r > 1:
    midp = (raices[0:(num_r-1)]+raices[1:num_r])/2
  else:
    midp = np.array([])
    
  if num_r > 0:
    ex1 = np.array([raices[0]-1])
    ex2 = np.array([raices[-1]+1])
    samples = np.concatenate([ex1, midp, ex2])
    temp = [(rawSVI(samples[i],a1,b1,r1,m1,s1)-rawSVI(samples[i],a2,b2,r2,m2,s2)) for i in range(num_r)]
    c = np.max(np.maximum(0,np.array(temp)))
  
  return raices, c


def g(k,a,b,rho,m,sigma): 
  discr = np.sqrt((k-m)**2 + sigma**2)
  w = a + b *(rho*(k-m)+ discr) #derivadas de rawSVI
  dw = b*rho + b *(k-m)/discr
  d2w = b*(sigma**2)/(discr*discr*discr)
  return (1 - k*dw/w + dw*dw/4*(-1/w+k*k/(w*w)-4) +d2w/2)

def butterfly_arbitrage(setT):
    #Aca cuando la funcion baja de cero hay arbitraje.
    #Luego para una cantidad peso que tanto se aleja del cero
    #Para tener un cuantificacion del arbitraje 
    a,b,rho,m,sigma  = setT
    k= np.linspace(-2,2, 50)
    x = g(k,a,b,rho,m,sigma)
    return sum([np.maximum(0,-y) for y in x])

#####################

def sviSqrt(k,w0,rho,eta):
    #Parametrizacion ATM con parametros ro y eta. Definidos por una phi del tipo "power law" 
    return w0/2*(1+rho*eta/np.sqrt(w0)*k+np.sqrt((eta/np.sqrt(w0)*k+rho)**2+1-rho**2))

def computeW0(ivolData):
    #Varianza total ATM 
    expDates = np.sort(ivolData.TTM.unique())
    nT = len(expDates)
    w0 = np.zeros(nT)

    for slice in range(nT):
        t = expDates[slice]
        texp = ivolData.TTM.to_numpy()
        iv = ivolData.vi_bs.to_numpy()[texp==t]
        f = (ivolData.F.to_numpy())[1]
        stk = ivolData.Strike.to_numpy()[texp==t]
        k = np.log(stk/f)
        interp = interp1d(k, iv**2) 
        w0[slice] = t*interp(0)
        
    return w0

def sviSqrtFit(ivolData):
    # Todas las slices juntas fiteo los valores de ro y eta
    # Que son correspondientes a todos los maturities
    expDates = np.sort(ivolData.TTM.unique())
    nSlices = len(expDates)
    nrows = ivolData.shape[0]
    midV = []
    kk = []
    ww0 = []
    for slice in range(nSlices):
        t = expDates[slice]
        texp = ivolData.TTM.to_numpy()
        stk = ivolData.Strike.to_numpy()
        iv = ivolData.vi_bs.to_numpy()[texp==t]
        f = (ivolData.F.to_numpy())[1]
        
        k = np.log(stk[texp==t]/f)
        interp = interp1d(k, iv**2) 
        w0 = t*interp(0)
        ww0.append(w0)
        midV.append(iv**2)
        kk.append(k)    
    tcutoff = np.minimum(0.1,np.max(expDates))

    param = []
    
    x0 = [0.5, 0.9]
    lim = [[-0.95,0.95],[-10000,10000]]

    
    def obj (x):
        rho,eta = x
        tmp = 0
        for i in range(nSlices):
            t =expDates[i]
            if t>tcutoff:
                sviSqrtVar = sviSqrt(kk[i],ww0[i],rho,eta)/t
                tmp += np.sum((midV[i]-sviSqrtVar)**2)
            else:
                pass
        return(tmp)
        
    result = minimize(obj,x0,method='SLSQP', bounds=lim)
    rho,eta = [param for param in result.x]
    
    #Obtenidos los resultados se pasan a sus parametros rawSVI
    for i in range(nSlices): 
        w0r = ww0[i]
        a = w0r/2*(1-rho**2)
        gg = eta/np.sqrt(w0r)
        b = w0r/2*gg
        m = -rho/gg
        sigma = np.sqrt(1-rho**2)/gg

        ij = [a,b,rho,m,sigma]
        param.append(ij)
    
    return param

###############

def sviFitQR(ivolData,sviGuess,pF=100,bF=100,method='SLSQP'):
    # El ajuste propiamente. Define los precios y calibra con ellos
    # Saliendo de w(k,t), aplicando BSM
    r=0.03
    callVals = ivolData.mid.to_numpy()
    expDates = np.sort(ivolData.TTM.unique())
    nSlices = len(expDates)
    slic = np.array(range(nSlices))[::-1]
    sviMatrix = sviGuess.copy()

    for i in slic: #Para cada slice corrijo individualmente
        temp = []
        t = expDates[i]
        texp = ivolData.TTM.to_numpy()

        midVal = callVals[texp==t]
        stk = ivolData.Strike.to_numpy()[texp==t]
        iv = ivolData.vi_bs.to_numpy()[texp==t]
        S0 = (ivolData.Spot.to_numpy())[1]
        f = (ivolData.F.to_numpy())[1]
        k = np.log(stk/f)

        def sqDist(x):

            a,b,rho,m,sigma  = x

            delta,mu,rho,omega,zeta = raw2nat(a,b,rho,m,sigma)
            
            sviVar = naturalSVI(k,delta,mu,rho,omega,zeta)/t
            #sviVar = rawSVI(k,a,b,rho,m,sigma)/t

            sviIV = np.sqrt(sviVar)

            outVal = [opcion_europea_bs('C',S0,stk[i],t,r,sviIV[i],0) for i in range(len(stk))] 
            outVal = np.array(outVal)
            tmp = np.sum((midVal-outVal)**2)
    
            return tmp
        
        
        
        def sqDistN(x): #Cuadrados minimos reescaleado por la adivinanza inicial
            return sqDist(x)/sqDist(sviGuess[i])

            
        def crossPen(x): #crossedness penalty
            a,b,rho,m,sigma  = x

            cPenalty = 0

            if i > 0: # Compara la de antes T
                cPenalty = np.real(calendar_arbitrage(x,sviMatrix[i-1])[1])
            else: # Caso i = 0
                minVar = a+b*sigma*np.sqrt(np.abs(1-rho**2))
                negVarPenalty = np.minimum(100,np.exp(-1/minVar))
                cPenalty = negVarPenalty

            if i < (nSlices-1): #Compara con la siguiente T, dado que comparo con la anterior primero
                cPenalty = cPenalty + np.real(calendar_arbitrage(x,sviMatrix[i+1])[1])
            
            return cPenalty*pF #factor de penalidad por c
        
        def buttPen(x):
            bPenalty = butterfly_arbitrage(x)
            return bPenalty*bF #factor de penalidad por butterfly

        
        def obj(x): #funcion total a minimizar
            return sqDistN(x)+crossPen(x)+buttPen(x)
        
        #a,b,rho,m,sigma 
        bound = [[-1000,1000],[1e-9,100],[-0.95,0.95],[-10,10],[1e-9,100]]
        if method == 'Nelder-Mead':
            result = minimize(obj,sviGuess[i],method='Nelder-Mead')
        elif method == 'SLSQP':
            result = minimize(obj,sviGuess[i],method='SLSQP', bounds=bound)#method='l-bfgs-b', bounds=bound)#
        
        sviMatrix[i]= [x for x in result.x]
    return sviMatrix