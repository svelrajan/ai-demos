# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Read the Products dataset
entity = pd.read_csv('https://raw.githubusercontent.com/svelrajan/saro-nw/main/Products-Manufactured.csv',
					index_col ='Month',
					parse_dates = True)

# Plot the raw datapoints (1st plot)
entity['#Products'].plot(figsize = (12, 5), legend = True)
plt.show()

#Seasonal Auto-Regressive Integrated Moving Average with eXogenous (exeternal cause/origin) factors
# Train the model on the full dataset
model = SARIMAX(entity['#Products'],
                        order = (0, 1, 1),
                        seasonal_order =(2, 1, 1, 12))

# This is where the ML Training happens
model_fit_result = model.fit()

# Forecast for the next 3 years
forecast = model_fit_result.predict(start = len(entity),
                          end = (len(entity)-1) + 3 * 12,
                          typ = 'levels').rename('Forecast')

# Plot the forecast values (2nd plot)
entity['#Products'].plot(figsize = (12, 5), legend = True)
forecast.plot(legend = True)
plt.show()
