'''
Created on May 29, 2017

@author: Yunshi_Zhao
'''
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
#import math


def acquire_data(file):
    """
    Convert csv file to pandas dataframe
    """
    
    return pandas.read_csv(file)


def explore_data(df):
    """
    Explore data type and range
    """
    
    df.info()
    print(df.head())
    print(df.tail())
    print(df.describe())
    
    
def clean_data(df):
    """
    Only keep useful information and delete duplicate information
    Convert to ordinal data for daily average temperature and daily average wind speed
    """
    
    df = df[['UNIT', 'DATEn', 'TIMEn', 'ENTRIESn_hourly', 'datetime', 'hour', 'day_week', 'weekday',
              'station', 'rain', 'fog', 'meanprecipi', 'meantempi', 'meanwspdi']]
    df=df.rename(columns = {'ENTRIESn_hourly':'Entries_per_4_hours'})
    
    df.loc[df['meantempi'] < 60, 'meantempi'] = 1
    df.loc[(df['meantempi'] >= 60) & (df['meantempi'] < 70), 'meantempi'] = 2
    df.loc[df['meantempi'] >= 70, 'meantempi'] = 3
    
    df.loc[df['meanwspdi'] < 4, 'meanwspdi'] = 1
    df.loc[(df['meanwspdi'] >= 4) & (df['meanwspdi'] < 6), 'meanwspdi'] = 2
    df.loc[(df['meanwspdi'] >= 6) & (df['meanwspdi'] < 8), 'meanwspdi'] = 3
    df.loc[df['meanwspdi'] >= 8, 'meanwspdi'] = 4
    return df


def rain_effect(dataframe): 
    """
    Compare entries distribution between rainy days and non-rainy days
    Perform Mann Whitney Test to exam the null hypothesis
    Return mean for both distribution and probability to accept the hypothesis
    """
    
    #Plot distribution
    plot = sns.FacetGrid(dataframe, col='rain')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    plt.subplots_adjust(top=0.9)
    #plot.fig.suptitle('Compare Entries Distribution on Rainy Days and Non Rainy Days')
    axes = plot.axes.flatten()
    axes[0].set_title("No Rain")
    axes[1].set_title("Rain")
    
    #Mann Whitney Test
    nrain_mean, rain_mean, U, p = mann_whitney_plus_means(dataframe[dataframe['rain'] == 0]['Entries_per_4_hours'], 
                                                          dataframe[dataframe['rain'] == 1]['Entries_per_4_hours'])
    
    return nrain_mean, rain_mean, U, p
    
    
def fog_effect(dataframe):
    """
    Compare entries distribution between foggy days and non-foggy days
    Perform Mann Whitney Test to exam the null hypothesis
    Return mean for both distribution and probability to accept the hypothesis
    """
    
    #Plot distribution
    plot = sns.FacetGrid(dataframe, col='fog')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    #plot.fig.suptitle('Compare Entries Distribution on foggy Days and Non foggy Days')
    axes = plot.axes.flatten()
    axes[0].set_title("No Fog")
    axes[1].set_title("Fog")
    
    #Mann Whitney Test
    nfog_mean, fog_mean, U, p = mann_whitney_plus_means(dataframe[dataframe['fog'] == 0]['Entries_per_4_hours'], 
                                                        dataframe[dataframe['fog'] == 1]['Entries_per_4_hours'])
    return nfog_mean, fog_mean, U, p


def rain_and_day_effect(dataframe):
    """
    Compare entries distribution between days of a week
    Perform Mann Whitney Test to exam the null hypothesis
    Return mean for both distribution and probability to accept the hypothesis
    """
    
    #day_week = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    #Plot distribution
    plot = sns.FacetGrid(dataframe, col='rain', row='weekday')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    axes[0].set_title("Rainy Weekend")
    axes[1].set_title("Non-Rainy Weekend")
    axes[2].set_title("Rainy Weekday")
    axes[3].set_title("Non-Rainy Weekday")
    
    """
    for i in range(len(axes)):
        if i%2 == 0:
            title = day_week[i//2] + "; Rain" 
        else: 
            title = day_week[i//2] + "; No Rain"  
        axes[i].set_title(title)
    """
    #Mann Whitney Test
    nrain_mean, rain_mean, U, p = [], [], [], []
    for i in range(2):
        daily_nrm, daily_rm, daily_U, daily_p = mann_whitney_plus_means \
        (dataframe[(dataframe['rain'] == 0) & (dataframe['weekday'] == i)]['Entries_per_4_hours'], 
         dataframe[(dataframe['rain'] == 1) & (dataframe['weekday'] == i)]['Entries_per_4_hours'])
        nrain_mean.append(daily_nrm)
        rain_mean.append(daily_rm)
        U.append(daily_U)
        p.append(daily_p)
    return nrain_mean, rain_mean, U, p
        
    
def weekday_effect(dataframe):
    """
    Compare entries distribution between weekday and weekend
    Perform Mann Whitney Test to exam the null hypothesis
    Return mean for both distribution and probability to accept the hypothesis
    """
    
    #Plot distribution
    plot = sns.FacetGrid(dataframe, col='weekday')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    axes[0].set_title("Weekday")
    axes[1].set_title("Weekend")
    
    #Mann Whitney Test
    weekday_mean, weekend_mean, U, p = mann_whitney_plus_means \
    (dataframe[dataframe['weekday'] == 1]['Entries_per_4_hours'], 
     dataframe[dataframe['weekday'] == 0]['Entries_per_4_hours'])
    return weekday_mean, weekend_mean, U, p


def time_effect(dataframe):
    """
    Compare entries distribution between days of a week
    Computer average entries for each time slot and plot them on a scatter plot
    """
    
    hour = ['12am', '4am', '8am', '12pm', '4pm', '8pm']
    #Plot distribution
    plot = sns.FacetGrid(dataframe, col='hour')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    for i in range(len(axes)):
        axes[i].set_title(hour[i])
    
    #Compare average entries
    avg_by_time = dataframe[['hour', 'Entries_per_4_hours']].groupby(['hour'], as_index=False).sum().sort_values(by='hour')
    plt.figure()
    plt.scatter(avg_by_time['hour'],avg_by_time['Entries_per_4_hours'])
    plt.xticks(avg_by_time['hour'],hour)


def temp_effect(dataframe):
    """
    Compare entries distribution between different temperature range
    Perform Mann Whitney Test to exam the null hypothesis
    Return mean for both distribution and probability to accept the hypothesis
    """
    temp = ['Under 60 degrees F', '60-70 degrees F', 'above 70 degrees F']
    
    #Plot distribution
    plot = sns.FacetGrid(dataframe, col='meantempi')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    for i in range(len(axes)):
        axes[i].set_title(temp[i])
    
    #Compare average entries    
    avg_by_temp = dataframe[['meantempi', 'Entries_per_4_hours']].groupby(['meantempi'], as_index=False).mean().sort_values(by='meantempi')
    plt.figure()
    plt.scatter(avg_by_temp['meantempi'],avg_by_temp['Entries_per_4_hours']) 
    plt.xticks(avg_by_temp['meantempi'],temp)
    
    #Mann Whitney Test
    chill_day_mean, hot_day_mean, U, p = mann_whitney_plus_means \
    (dataframe[dataframe['meantempi'] < 2]['Entries_per_4_hours'], 
     dataframe[dataframe['meantempi'] == 2]['Entries_per_4_hours'])
    return chill_day_mean, hot_day_mean, U, p

    
def wind_effect(dataframe):
    """
    Compare entries distribution between different wind speed
    Perform Mann Whitney Test to exam the null hypothesis
    Return mean for both distribution and probability to accept the hypothesis
    """
    
    wsp = ['under 4 mph', '4-6 mph', '6-8 mph', 'above 8 mph']
    
    #Plot distribution
    plot = sns.FacetGrid(dataframe, col='meanwspdi')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    for i in range(len(axes)):
        axes[i].set_title(wsp[i])
    
    #Compare average entries     
    avg_by_wind = dataframe[['meanwspdi', 'Entries_per_4_hours']].groupby(['meanwspdi'], as_index=False).mean().sort_values(by='meanwspdi')
    plt.figure()
    plt.scatter(avg_by_wind['meanwspdi'],avg_by_wind['Entries_per_4_hours']) 
    plt.xticks(avg_by_wind['meanwspdi'],wsp)  
    
    #Mann Whitney Test
    normal_day_mean, windy_day_mean, U, p = mann_whitney_plus_means \
    (dataframe[dataframe['meanwspdi'] < 4]['Entries_per_4_hours'], 
     dataframe[dataframe['meanwspdi'] == 4]['Entries_per_4_hours'])
    return normal_day_mean, windy_day_mean, U, p


def mann_whitney_plus_means(dist1, dist2):
    """
    Perform Mann Whitness Test for Non-Normally Distributed data
    """
    
    mean1 = np.mean(dist1)
    mean2 = np.mean(dist2)
    U,p = scipy.stats.mannwhitneyu(dist1, dist2)
    return mean1, mean2, U, p


def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma


def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, 
    and the values for our thetas.
    """
    
    cost = np.square(np.dot(features,theta)-values).sum()
    return cost


def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.

    """
    
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        theta = theta - alpha/m*np.dot(np.dot(features,theta)-values,features)
        cost = np.square(np.dot(features,theta)-values).sum()
        cost_history.append(cost)
        
    return theta, pandas.Series(cost_history)


def predictions(dataframe, features):
    """
    Find linear regression model using gredient decent
    """
    
    # Select Features
    features_df = dataframe[features]

    # Add station to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['station'], prefix='station')
    features_df = features_df.join(dummy_units)
    dummy_units = pandas.get_dummies(dataframe['hour'], prefix='hour')
    features_df = features_df.join(dummy_units)

    # Values
    values = dataframe['Entries_per_4_hours']
    m = len(values)
    

    features_df, mu, sigma = normalize_features(features_df)
    features_df['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features_df)
    values_array = np.array(values)
    
    #Perform transformation
    #values_array = log_transform(values_array)


    # Set values for alpha, number of iterations.
    alpha = 0.5 # please feel free to change this value
    num_iterations = 40 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features_df.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    # Plot cost history
    plot_cost_history(alpha, cost_history)  
    predictions = np.dot(features_array, theta_gradient_descent)
    
    
    #predictions = np.exp(predictions)

    
    return predictions


def predictions_Lasso(dataframe, features):
    """
    Find linear regression model using Lasso method
    """
    # Select Features
    features_df = dataframe[features]
        
    # Add station to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['station'], prefix='station')
    features_df = features_df.join(dummy_units)
    
    # Values
    values = dataframe['Entries_per_4_hours']
    m = len(values)

    features_df, mu, sigma = normalize_features(features_df)
    features_df['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features_df)
    values_array = np.array(values)
    
    #Lasso Linear Regression
    regr = linear_model.Lasso(alpha = 0.1,max_iter=2000)
    regr.fit(features_array, values_array)
    prediction = regr.predict(features_array)
    
    return prediction


def predictions_OLS(dataframe, features):
    """
    Find linear regression model using ordinary least squares
    """
    
    features_df = dataframe[features]
        
    # Add station to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['station'], prefix='station')
    features_df = features_df.join(dummy_units)
    
    # Values
    values = dataframe['Entries_per_4_hours']
    m = len(values)

    features_df, mu, sigma = normalize_features(features_df)
    features_df['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features_df)
    values_array = np.array(values)
    
    reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    reg.fit (features_array, values_array)
    
    
    prediction = reg.predict(features_array)
    return prediction


def plot_cost_history(alpha, cost_history):
    """This function is for viewing the plot of your cost history."""

    plt.figure()
    plt.plot(range(len(cost_history)),cost_history)


def plot_residuals(dataframe, predictions):
    '''
    a histogram of the residuals

    '''
    
    plt.figure()
    (dataframe['Entries_per_4_hours'] - predictions).hist(bins=500)
    plt.figure()
    plt.scatter(predictions, dataframe['Entries_per_4_hours'] - predictions)
    #return plt
    
def compute_r_squared(dataframe, predictions):
    '''
    calculate the R^2 value
    '''
    data = np.array(dataframe['Entries_per_4_hours'])
    ssr=(np.square(predictions - np.mean(predictions))).sum()
    sst=(np.square(data-np.mean(data))).sum()
    r_squared = ssr/sst
    return r_squared

def log_transform(x):
    """ 
    Transform an array from linear scale to logarithmic scale
    """
    x[x == 0] = 1
    return np.log(x)
# -------------------------------------------------

#file = 'C:/Users/Yunshi_Zhao/workspace/NYSubway/Data/turnstile_weather_v2.csv'
#nysw = acquire_data(file)
#explore_data(nysw)
#nysw = clean_data(nysw)
#explore_data(nysw)

#rain_effect(nysw)
#fog_effect(nysw)
#day_effect(nysw)
#weekday_effect(nysw)
#time_effect(nysw)
#temp_effect(nysw)
#wind_effect(nysw)
#plt.show()
#nysw = nysw[nysw['Entries_per_4_hours'] > 6000]



#nysw['rain_day'] = nysw['rain'] * nysw['weekday']
#nysw['meanwspdi_sq'] = nysw['meanwspdi'] ** 2
#nysw['hour_cb'] = nysw['hour'] ** 3
#features = ['weekday', 'rain', 'rain_day','meanwspdi_sq', 'meanprecipi', 'meanwspdi', 'meantempi']
#pred = predictions(nysw, features)
#plot_residuals(nysw, pred)
#print(compute_r_squared(nysw, pred))
#plt.show()
