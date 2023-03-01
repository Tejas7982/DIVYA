import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot 


df = pd.read_csv(r"C:\Users\Maitrayee\Desktop\allot plastic\cleaned_allot_Plastic_data.csv")



# Data Partition
Train = df.head(24)
Test = df.tail(12)


tsa_plots.plot_acf(df.QUANTITY, lags = 12)
tsa_plots.plot_pacf(df.QUANTITY, lags = 12)


# ARIMA 
model1 = ARIMA(Train.QUANTITY, order = (1, 1, 12))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
start_index
end_index = start_index + 11
forecast_test = res1.predict(start = start_index, end = end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.QUANTITY, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.QUANTITY)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm
help(pm.auto_arima)

ar_model = pm.auto_arima(Train.QUANTITY, start_p = 0, start_q = 0,
                      max_p = 16, max_q = 16, # maximum p and q
                      m = 1,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = False,   # No Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)


# Best Parameters ARIMA

model = ARIMA(Train.QUANTITY, order = (0,1,0))
res = model.fit()
print(res.summary())


# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res.predict(start = start_index, end = end_index)


print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.QUANTITY, forecast_best))
print('Test RMSE: %.3f' % rmse_best)
# plot forecasts against actual outcomes
pyplot.plot(Test.QUANTITY)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()


# checking both rmse of with and with out autoarima

print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)

res1.save("model.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

# Forecast for future 12 months
start_index = len(df)
end_index = start_index + 11
forecast = model.predict(start = start_index, end = end_index)

print(forecast)




