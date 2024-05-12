# Import required packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# To reduce the printing of the warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Importing models and tools from the statsmodels library
from statsmodels.tsa.stattools import adfuller  # test for stationarity
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # determine p and q arguments of ARIMA(p,q,d)
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Importing models and tools from the pmdarima package
from pmdarima.arima import auto_arima
from pmdarima.arima.utils import ndiffs

# Accuracy metrics from Scikit-learn to evauate model's performance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler



######################################################################################################

# A function to read in the data 
def read_in_data(file_name):
    """
    expected a filename as a string that will have a .csv extension
    """
    data = pd.read_csv(f'{file_name}', index_col = "timestamp", parse_dates = True).asfreq('D')
    return(data)



# A function to pre-process data so we can have a pandas Series for forcasting model
def prep_series_for_forecasting(data_df, feature_name):
    """
    * data_df - expected a data frame that contains the feature with numerial values 
    that will be used for timeseries forecasting 
    * name_of_feature - column name that will be used for timeseries forecasting
    * it is expecte that the index is set as dates
    """
    
    # Select the column/variable that will be used for forecasting
    df = data_df[[feature_name]]
    
    # Assign the series for forecasting to a new variable
    values = df[feature_name]
    
    return(df, values)
    
    
    
# A function that uses Augmented Dickey Fuller test to check whether the series is stationary
def stationarity_check(series):
    """
    * series - expected pandas Series object
    """
    
    # Checking whether series is stationary using Augmented
    # Dickey Fuller (ADF) test

    result = adfuller(series.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    
    # Using .ndiffs() method from pmdadima to determine the minimum number of differencing required to 
    # make data stationary
    d = ndiffs(series, test = 'adf')
    print(f'\nThe number of differencing required for this data to be stationary is {d}.')

    
# A function to difference the data so the data becomes stationary    
def differencing(series):
    # Differenced data
    prices_diff = series.diff().dropna()
    
    return(prices_diff)

    # # ADF test
    # adf_res_diff_data = adfuller(prices_diff)
    # print(f'ADF Statistic: {adf_res_diff_data[0]}')
    # print(f'p-value:  {adf_res_diff_data[1]}')

    
# A function to split the data into training and testing sets
def split_train_test(series, n):
    """
    * series - expected pandas Series object
    * n - a non-negative integer specifying how many predictions to be made
    """
    
    # Split into train and test datasets
    train = series[ : len(series) - n]
    test = series[len(series) - n : ]
    
    return(train, test)



# A function to implement Auto-Regressive (AR) model
def AR_forecasting(train, test, p):
    """
    * series - expected pandas Series object
    * p - a non-negative integer specifying the order of the AM component
    """
    
    # Define Auto-Regressive (AR) model from statsmodels (AutoReg)
    model = AutoReg(train, lags = p).fit()
    
    # Print summary of the model
    print(model.summary())
    
    # Make predictions
    predictions = model.predict(start = len(train), end = len(train) + len(test) - 1, dynamic = False)
    
    return(predictions)



# A function to implement Moving Average (MA) model
def MA_forecasting(train, test, q):
    """
    * series - expected pandas Series object
    * q - a non-negative integer specifying the order of the MA component
    """
    
    # Define Moving Average (MA) model from statsmodels
    model = ARIMA(train, order = (0, 0, 8))
    
    print('\n')
    
    # Fit the model
    model_fit = model.fit()
    
    # Print summary of the model
    print(model_fit.summary())

    # Make predictions
    predictions = model_fit.predict(start = len(train), end = len(train) + len(test) - 1)
    
    return(predictions)



# A function to implement Auto-Regressive Integrated Moving Average (ARIMA) model
def ARIMA_forecasting(train, test, p, d, q):
    """
    * series - expected pandas Series object
    * p - a non-negative integer specifying the order of the AR component
    * d - a non-negative integer specifying the order of the I component - 
      the number of time the series need to be differenced
    * q - a non-negative integer specifying the order of the MA component
    """
    
    # Define Auto-Regressive Integrated Moving Average (ARIMA) model from statsmodels
    model = ARIMA(train, order = (p, d, q))
    
    # Fit the model
    model_fit = model.fit()
    
    # Print summary of the model
    print(model_fit.summary())

    # Make predictions
    predictions = model_fit.predict(start = len(train), end = len(train) + len(test) - 1, dynamic = False)
    
    return(predictions)



# A function to implement Seasonal Auto-Regressive Integrated Moving Average (SARIMA) model
def SARIMA_forecasting(train, test, p, d, q, P, D, Q, m):
    """
    * series - expected pandas Series object
    * p - a non-negative integer specifying the order of the AR component
    * d - a non-negative integer specifying the order of the I component - 
      the number of time the series need to be differenced
    * q - a non-negative integer specifying the order of the MA component
    * P - a non-negative integer specifying the order of the seasonal AR component
    * D - a non-negative integer specifying the order of the seasonal I component - 
      the seasonal difference, e.g. m = 7, 12, 1 - daily, monthly and yearly respectively
    * Q - a non-negative integer specifying the order of the MA component
    """

    # Define Moving Average (MA) model from statsmodels
    model = SARIMAX(train, order = (p, d, q), seasonal_order = (P, D, Q, m))
    
    # Fit the model
    model_fit = model.fit()
    
    # Print summary of the model
    print(model_fit.summary())
    print('\n\n\n')

    # Make predictions
    predictions = model_fit.predict(start = len(train), end = len(train) + len(test) - 1, dynamic = False)
    
    return(predictions)
    
    
# A function to visualise train, test and predictions
def plot_train_test_predictions_ARIMA(series, train, test, predictions, company_name, model_name, p, d, q, fig_folder):
    """
    * series - expected pandas Series object
    * company_name - the name of the company for plotting purposes - expected list[i]
    * model_name - the name of the model used for plotting purposes - expected list[i]
    * chosen_model - one of the defined time series models in the notebook
    """

    plt.rcParams["figure.figsize"] = (9, 6)
    
    # Plot 1
    plt.plot(series['2023-01-01':], 'blue', label = 'data')
    # plt.plot(test, 'green', label = 'test data')
    plt.plot(predictions, 'red', label = 'predictions')

    leg = plt.legend(loc='upper left') 
    # leg.get_frame().set_alpha(0.01)

    plt.xticks(rotation = 90, ha = 'right')
    plt.title(f'Predictions for {company_name}\'s daily stock prices using {model_name}({p},{d},{q})', fontweight="bold")
    plt.xlabel('Date', fontsize = 13)
    plt.ylabel('Price', fontsize = 13)
    
    plt.tight_layout()

    plt.savefig(f'{fig_folder}/{model_name}_{p}_{d}_{q}_model_using_statsmodels_{company_name}_daily_returns.pdf')
    plt.show()
    

    
# A function to visualise train, test and predictions
def plot_train_test_predictions_SARIMA(series, train, test, predictions, company_name, model_name, p, q, d, P, Q, D, s, fig_folder):
    """
    * series - expected pandas Series object
    * company_name - the name of the company for plotting purposes - expected list[i]
    * model_name - the name of the model used for plotting purposes - expected list[i]
    * chosen_model - one of the defined time series models in the notebook
    * n - number of data points for prediction
    """

    plt.rcParams["figure.figsize"] = (9, 6)
    
    # Plot 1
    plt.plot(series['2023-01-01':], 'blue', label = 'data')
    # plt.plot(test, 'green', label = 'test data')
    plt.plot(predictions, 'red', label = 'predictions')

    leg = plt.legend(loc='best', ncol=3) 
    # leg.get_frame().set_alpha(0.01)

    plt.xticks(rotation = 90, ha = 'right')
    plt.title(f'Predictions for {company_name}\'s daily stock prices using {model_name}({p}, {q}, {d})({P}, {Q}, {D}, {s})', fontweight="bold")
    plt.xlabel('Date', fontsize = 13)
    plt.ylabel('Price', fontsize = 13)
    
    plt.tight_layout()

    plt.savefig(f'{fig_folder}/{model_name}_({p}_{q}_{d})_({P}_{Q}_{D}_{s})_model_using_statsmodels_{company_name}_daily_returns.pdf')
    plt.show()
    

    
def model_eval(train, test, predictions, company_name, model_name):
    
    print(f'The evaluation of the given {model_name} model:')
    
    score = r2_score(test, predictions)
    mse = mean_squared_error(test, predictions)

    print("\n R^2 score is: {:.6f}".format(score))
    print("\n The MSE is: {:.6f}".format(mse))