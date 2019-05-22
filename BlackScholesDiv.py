# NOMBRE: Carlos Santillán
# FECHA: 2015
# DESC.: Black Scholes (con pago de dividendos)
#######################################################################################################################

## Importamos paquetes

import numpy as np
import scipy.stats as si

#######################################################################################################################

## Cuando hay pago de dividendos, sólo consideramos el nueva parámetro "q"
## Implementamos función para opción que paga dividendos, se escoge entre "PUT" Y "CALL"
def euro_vanilla_dividend(S, K, T, r, q, sigma, option='call'):
    # S: Precio Spot
    # K: Precio Strike
    # T: Madurez
    # r: Tasa de interés
    # q: Tasa de pago de dividendos (continua)
    # sigma: volatilidad del subyacente

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option == 'call':
        result = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))

    return result
#######################################################################################################################

## Función para calcular una "Vanilla Option" sin pago de dividendos
def euro_vanilla(S, K, T, r, sigma, option='call'):
    # S: Precio Spot
    # K: Precio Strike
    # T: Madurez
    # r: Tasa de Interés
    # sigma: Volatilidad del subyacente

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option == 'call':
        result = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    return result


