# volatility_surface
Superficie de Volatilidad Implícita - Grupo 3 - Análisis de Finanzas Cuantitativas (FCEN UBA)

Construcción de una grilla arbitraria de maturities y strikes para generar una superficie de volatilidad implicita que este calibrada a las volatilidades de un conjunto discreto de opciones de mercado. Esto se hace aplicando modelado estocastico para el subyacente (Modelos de Heston y Heston con Saltos, mediante priiceo por inversa de Fourier), como aplicando un modelo parametrico (SVI)

### Descripcion:

- `arbitrage.py`: Chequeo de arbitraje aplicado a datos de mercado
- `datosSchoutens`: Datos para correr pruebas, sacados del paper de Schoutens et al. "A perfect calibration. Now what?"
- `datoscallsSPX.csv`: csv de datos de calls de SPX recopilados con yfinance
- `datoscallsSPX_noArb.py`: csv de los datos anteriores habiendole hecho el chequeo de arbitraje previo
- `impvol.py`: Modulo para calcular volatilidad implicita
- `opcion_europea_bs.py`: Formula cerrada de Black Scholes para opciones europeas
- `pricers.py`: Modulo para calcular precios en el modelo de Heston y de Heston con saltos mediante inversion de Fourier
- `svi.py`: Funciones para aplicar el modelo de SVI
- `volatilidad_implicita_G3.ipynb`: Cuaderno de aplicación de lo anterior
