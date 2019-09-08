
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, YearLocator, DateFormatter
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
        
register_matplotlib_converters()

df = pd.read_csv('TSLA.csv')

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)



size_partition = (int)(len(df)*0.2) # partion test 20%, train 80%
df_test = df[-size_partition:]
print(df_test.shape)
df_train = df[:-size_partition]
print(df_train.shape)

#Graph start
fig, ax = plt.subplots()
ax.grid(True)
ax.xaxis.set_major_locator(YearLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y'))
ax.autoscale_view()
fig.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('Price')
fig.suptitle  ('Stock Price')
plt.plot(df_train['Adj Close'], label='Train')
plt.plot(df_test['Adj Close'], label ='Test')
plt.legend()
plt.show()
#Graph end


#Feature Engineering


# Get indices of access for the data
window_size=7
num_samples = len(df) - window_size
indices = np.arange(num_samples).astype(np.int)[:,None] + np.arange(window_size + 1).astype(np.int)

data = df['Adj Close'].values[indices] # Create the 2D matrix of training samples

X = data[:,:-1] # Each row represents 32 days in the past
y = data[:,-1] # Each output value represents the 33rd day

# Train and test split
split_fraction = 0.8
ind_split = int(split_fraction * num_samples)
X_train = X[:ind_split]
y_train = y[:ind_split]
X_test = X[ind_split:]
y_test = y[ind_split:]

# Train
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Infer
y_pred_train_ridge = ridge_model.predict(X_train)
y_pred_ridge = ridge_model.predict(X_test)

#Plot
df_ridge = df.copy()
df_ridge.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_ridge = df_ridge.iloc[ind_split+window_size:]
df_ridge['Adj Close Test'] = y_pred_ridge


#Graph start
fig.suptitle  ('Ridge Model')
plt.plot(df_ridge['Adj Close'],label='Adj Close')
plt.plot(df_ridge['Adj Close Test'], label='Adj Close Test')
plt.show()
#Graph end


# Model #2 - Gradient Boosting Trees
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# Infer
y_pred_train_gb =gb_model.predict(X_train)
y_pred_gb = gb_model.predict(X_test)

df_gb = df.copy()
df_gb.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_gb = df_gb.iloc[ind_split+window_size:] 
df_gb['Adj Close Test'] = y_pred_gb


#Graph 
fig.suptitle  ('Gradient Boost')
plt.plot(df_gb['Adj Close'],label='Adj Close')
plt.plot(df_gb['Adj Close Test'], label='Adj Close Test')
plt.show()
#Graph end

