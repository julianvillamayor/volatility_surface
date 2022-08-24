import pandas as pd
import numpy as np

def check_calendar_spread(data):
    stk = np.sort(data.strike.unique())
    arb_indices = []
    offset = []
    for elm in stk:
        temp = data[data['strike']==elm]
        ttm = np.sort(temp.TTM.unique())
        temp = temp.sort_values(by=['TTM'])
        indices = list(temp.index.values)
        precios = temp.mid.to_numpy()
        nT = len(ttm)
        if nT <2:
            pass
        else:
            for i in range(len(indices)-1):
                diff = precios[i+1] - precios[i]
                if  diff < 0:
                    arb_indices.append(indices[i])
                    offset.append(diff)
    
    return arb_indices,offset

def check_vertical_spread(data):
    ttm = np.sort(data.TTM.unique())
    arb_indices = []
    offset = []
    for elm in ttm:
        temp = data[data['TTM']==elm]
        stk = np.sort(temp.strike.unique())
        temp = temp.sort_values(by=['strike'])
        indices = list(temp.index.values)
        precios = temp.mid.to_numpy()
        nK = len(stk)
        if nK <2:
            pass
        else:
            
            for i in range(len(indices)-1):
                diff = precios[i] - precios[i+1]
                if  diff < 0:
                    arb_indices.append(indices[i])
                    offset.append(diff)
    
    return arb_indices,offset


def check_butterfly_spread(data):
    ttm = np.sort(data.TTM.unique())
    arb_indices = []
    offset = []
    for elm in ttm:
        temp = data[data['TTM']==elm]
        stk = np.sort(temp.strike.unique())
        temp = temp.sort_values(by=['strike'])
        indices = list(temp.index.values)
        precios = temp.mid.to_numpy()
        nK = len(stk)
        if nK <2:
            pass
        else:
            for i in range(len(indices)-2):
                a = (stk[i+2]-stk[i])/(stk[i+1]-stk[i])
                diff = precios[i] - a*precios[i+1] + precios[i+2]
                if  diff < 0:
                    arb_indices.append(indices[i])
                    offset.append(diff)
    
    return arb_indices,offset

def price_bounds(data,S0):
    arb_indices= []
    ttm = np.sort(data.TTM.unique())
    for elm in ttm:
        temp = data[data['TTM']==elm]
        temp = temp.sort_values(by=['strike'])
        indices = list(temp.index.values)
        k = temp.strike.to_numpy()
        for i in range(len(indices)):
            precio = temp.mid.to_numpy()[k==k[i]]
            cond = (np.max(S0-k[i],0) < precio) & (precio < S0)
            if cond == False:
                arb_indices.append()
    
    return arb_indices

