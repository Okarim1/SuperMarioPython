# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:53:29 2018

@author: Okarim
"""

import numpy as np

def bottomToTop(texto):
    """
    texto=nivel en formato de texto
    secuencia= secuencia de caracteres obtenidos al recorrer el nivel de abajo hacia arriba
    """
    secuencia=[]
    for i in range(len(texto[0])):
        for j in reversed(range(len(texto))):
            secuencia.append(texto[j][i])
    return secuencia

def snaking(texto):
    """
    texto=nivel en formato de texto
    secuencia= secuencia de caracteres obtenidos al recorrer el nivel serpenteando
    """
    secuencia=[]
    for i in range(len(texto[0])):
        for j in range(len(texto)):
            if(i%2==0):
                secuencia.append(texto[j][i])
            else:
                secuencia.append(texto[len(texto)-j-1][i])
    return secuencia

if __name__ == '__main__':
    texto=np.loadtxt("Levels/mario-1-1.txt", dtype=str, comments="~")
    
    sec=bottomToTop(texto)
    print(sec[:30])
    
    sec=snaking(texto)
    print(sec[:30])