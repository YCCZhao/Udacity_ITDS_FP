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
from sklearn.model_selection import cross_val_predict


def acquire_data(file):
    return pandas.read_csv(file)

def explore_data(df):
    df.info()
    print(df.head())
    print(df.tail())
    print(df.describe())
    
def clean_data(df):
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
    #df['meantempi'] = df['meantempi'].astype(int)
    #df['meanwspdi'] = df['meanwspdi'].astype(int)
    return df

def rain_effect(dataframe): 
    plot = sns.FacetGrid(dataframe, col='rain')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    axes[0].set_title("No Rain")
    axes[1].set_title("Rain")
     
    nrain_mean, rain_mean, U, p = mann_whitney_plus_means(dataframe[dataframe['rain'] == 0]['Entries_per_4_hours'], 
                                                          dataframe[dataframe['rain'] == 1]['Entries_per_4_hours'])
    return nrain_mean, rain_mean, U, p
    
def fog_effect(dataframe):
    plot = sns.FacetGrid(dataframe, col='fog')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    axes[0].set_title("No Fog")
    axes[1].set_title("Fog")
    nfog_mean, fog_mean, U, p = mann_whitney_plus_means(dataframe[dataframe['fog'] == 0]['Entries_per_4_hours'], 
                                                        dataframe[dataframe['fog'] == 1]['Entries_per_4_hours'])
    return nfog_mean, fog_mean, U, p


def day_effect(dataframe):
    day_week = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    plot = sns.FacetGrid(dataframe, col='rain', row='day_week')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    for i in range(len(axes)):
        if i%2 == 0:
            title = day_week[i//2] + "; Rain" 
        else: 
            title = day_week[i//2] + "; No Rain"  
        axes[i].set_title(title)
    nrain_mean, rain_mean, U, p = [], [], [], []
    for i in range(len(day_week)):
        daily_nrm, daily_rm, daily_U, daily_p = mann_whitney_plus_means \
        (dataframe[(dataframe['rain'] == 0) & (dataframe['day_week'] == i)]['Entries_per_4_hours'], 
         dataframe[(dataframe['rain'] == 1) & (dataframe['day_week'] == i)]['Entries_per_4_hours'])
        nrain_mean.append(daily_nrm)
        rain_mean.append(daily_rm)
        U.append(daily_U)
        p.append(daily_p)
    return nrain_mean, rain_mean, U, p
        
    
def weekday_effect(dataframe):
    plot = sns.FacetGrid(dataframe, col='weekday')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    axes[0].set_title("Weekday")
    axes[1].set_title("Weekend")
    weekday_mean, weekend_mean, U, p = mann_whitney_plus_means \
    (dataframe[dataframe['weekday'] == 1]['Entries_per_4_hours'], 
     dataframe[dataframe['weekday'] == 0]['Entries_per_4_hours'])
    return weekday_mean, weekend_mean, U, p

def time_effect(dataframe):
    hour = ['12am', '4am', '8am', '12pm', '4pm', '8pm']
    plot = sns.FacetGrid(dataframe, col='hour')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    for i in range(len(axes)):
        axes[i].set_title(hour[i])
    
    avg_by_time = dataframe[['hour', 'Entries_per_4_hours']].groupby(['hour'], as_index=False).sum().sort_values(by='hour')
    plt.figure()
    plt.scatter(avg_by_time['hour'],avg_by_time['Entries_per_4_hours'])
    plt.xticks(avg_by_time['hour'],hour)


def temp_effect(dataframe):
    temp = ['Under 60 degrees F', '60-70 degrees F', 'above 70 degrees F']
    plot = sns.FacetGrid(dataframe, col='meantempi')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    for i in range(len(axes)):
        axes[i].set_title(temp[i])
        
    avg_by_temp = dataframe[['meantempi', 'Entries_per_4_hours']].groupby(['meantempi'], as_index=False).mean().sort_values(by='meantempi')
    plt.figure()
    plt.scatter(avg_by_temp['meantempi'],avg_by_temp['Entries_per_4_hours']) 
    plt.xticks(avg_by_temp['meantempi'],temp)
    chill_day_mean, hot_day_mean, U, p = mann_whitney_plus_means \
    (dataframe[dataframe['meantempi'] < 2]['Entries_per_4_hours'], 
     dataframe[dataframe['meantempi'] == 2]['Entries_per_4_hours'])
    return chill_day_mean, hot_day_mean, U, p
    
def wind_effect(dataframe):
    wsp = ['under 4 mph', '4-6 mph', '6-8 mph', 'above 8 mph']
    plot = sns.FacetGrid(dataframe, col='meanwspdi')
    plot.map(plt.hist, 'Entries_per_4_hours', range=[0,6000])
    axes = plot.axes.flatten()
    for i in range(len(axes)):
        axes[i].set_title(wsp[i])

    avg_by_wind = dataframe[['meanwspdi', 'Entries_per_4_hours']].groupby(['meanwspdi'], as_index=False).mean().sort_values(by='meanwspdi')
    plt.figure()
    plt.scatter(avg_by_wind['meanwspdi'],avg_by_wind['Entries_per_4_hours']) 
    plt.xticks(avg_by_wind['meanwspdi'],wsp)  
    normal_day_mean, windy_day_mean, U, p = mann_whitney_plus_means \
    (dataframe[dataframe['meanwspdi'] < 3]['Entries_per_4_hours'], 
     dataframe[dataframe['meanwspdi'] == 4]['Entries_per_4_hours'])
    return normal_day_mean, windy_day_mean, U, p


def mann_whitney_plus_means(dist1, dist2):
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

    # Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to change this value
    num_iterations = 75 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features_df.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    plot_cost_history(alpha, cost_history)  
    predictions = np.dot(features_array, theta_gradient_descent)
    return predictions


def predictions_Lasso(dataframe, features):
    
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
    #LinearRegression()
    prediction = reg.predict(features_array)
    return prediction

def plot_cost_history(alpha, cost_history):
    """This function is for viewing the plot of your cost history."""

    plt.figure()
    plt.plot(cost_history, range(len(cost_history)))

def plot_residuals(dataframe, predictions):
    '''
    a histogram of the residuals

    '''
    
    plt.figure()
    (dataframe['Entries_per_4_hours'] - predictions).hist(bins=500)
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
# -------------------------------------------------

file = 'C:/Users/Yunshi_Zhao/workspace/NYSubway/Data/turnstile_weather_v2.csv'
nysw = acquire_data(file)
#explore_data(nysw)
nysw = clean_data(nysw)
#explore_data(nysw)

#rain_effect(nysw)
#fog_effect(nysw)
#day_effect(nysw)
#weekday_effect(nysw)
#time_effect(nysw)
#temp_effect(nysw)
#wind_effect(nysw)
#plt.show()

features = ['rain', 'meanwspdi', 'hour', 'meantempi']
pred = predictions_OLS(nysw, features)
plot_residuals(nysw, pred)
plt.show()
print(compute_r_squared(nysw, pred))