import yfinance as yf
import numpy as np 
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
yf.pdr_override() 
import datetime as dt 
from sklearn.cluster import KMeans
from functools import reduce
from dateutil.relativedelta import relativedelta
plt.style.use('fivethirtyeight')
import io

# The assets formula allow to the user access to the different assets in markets like the Dow Jones, the Standard and poor's 500 and the Australian Share Market 200

def assets(market):
#With this function the user can pick between three different indexes 'DJI', 'S&P500' and 'ASX  and work with the model 
    if market == 'DJI':
        table = pd.read_html('https://www.investopedia.com/terms/d/djia.asp')[0]
        tickers = table[('Dow Jones Industrial Average Components',    'Symbol')].tolist()
    elif market == 'S&P500':
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = table['Symbol'].tolist()
    elif market == 'ASX':
        table = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[0]
        table['Code'][:3]
        tickers = []
        for asset in table['Code']:
            tickers.append(asset+'.AX')
    return tickers


def getdata_kmeans(tickers,start,end):

#load the data in one DataFrame
    ind_data = pd.DataFrame()
    for t in tickers:
        ind_data[t] = pdr.DataReader(t,data_source='yahoo', start= start, end =end )['Adj Close']
    #print (ind_data.head())


#calculate the annual returns and variance
    daily_returns = ind_data.pct_change()
    annual_mean_returns = daily_returns.mean()*251
    annual_returns_variance = daily_returns.var()* 251

#create a new dataframe
    df = pd.DataFrame(ind_data.columns, columns = ['Stock Symbols'])
    df['Variances'] = annual_returns_variance.values
    df['Returns'] = annual_mean_returns.values

#Check if we have miss data 
    if df.isna().values.any().sum()> 0:
        check_missing = df.isnull()
        for column in check_missing.columns.values.tolist():
            df.dropna(axis = 0, inplace= True)
    
    return df

# transform the plot to png format
def plt_to_np(fig):
    with io.BytesIO() as buff:
        #fig.savefig(buff, format='png')
        plt.savefig(buff, format='png')
        buff.seek(0)
        im =plt.imread(buff)
    return im

def elbow_method (data, random_state=99):

#Load the data 
    X= data[['Returns', 'Variances']].values
    inertia_list = []
    for k in range(2,25):

#create and train the model
        kmeans = KMeans(n_clusters = k, random_state=random_state)
        kmeans.fit(X)
        inertia_list.append(kmeans.inertia_)

#Plot the data
    f1 = plt.figure()
    plt.plot(range(2,25), inertia_list)
    plt.title('Elbow Curve')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia or Sum Squared Error(SSE)')
    plot = plt_to_np(f1)
   

#Get and Show the labels / groups
    kmeans = KMeans(n_clusters = 5).fit(X)
    labels = kmeans.labels_
    data['Cluster_labels'] = labels
    return [plot , data]
    
    
def besteachclust (data):

#Find the best asset given each cluster we create 5 different cases 
    c = []
    for i in range(0,5):
        a = data.loc[(data['Cluster_labels'] == i)].sort_values(by="Returns",ascending=False).index[0]
        c.append(a)
    dfa= data.loc[c, :]
    df2 = dfa.sort_values(by="Returns",ascending=False)
    df2['Cluster_legend']= ['Really good Positive Returns','Medium Returns', 'Acceptable Positive Returns','Close to zero Returns','Negative Returns']
    
# Merge information in one data set to be used after for plot
    legend = df2[['Cluster_labels','Cluster_legend']]
    legend.set_index('Cluster_labels')
    test3 = pd.merge(data,legend, on ='Cluster_labels')
    return test3
    
  
def plotall(data):
#Use the right information to plot the differents clusters

#Create the different dataframes for every case 
    legend = ['Really good Positive Returns','Medium Returns', 'Acceptable Positive Returns','Close to zero Returns','Negative Returns']

    Good_returns = data.loc[(data['Cluster_legend']== 'Really good Positive Returns')]
    Medium_Returns = data.loc[(data['Cluster_legend']== 'Medium Returns')]
    Acceptable_Returns = data.loc[(data['Cluster_legend']== 'Acceptable Positive Returns')]
    Closezero = data.loc[(data['Cluster_legend']== 'Close to zero Returns')]
    Negative_Returns = data.loc[(data['Cluster_legend']== 'Negative Returns')]

#Plotting the  clusters    
    plt.figure(figsize=(9,9), dpi=80)

    colors = ['b', 'c', 'y', 'm', 'r']
    Good = plt.scatter(Good_returns['Returns'], Good_returns['Variances'],   marker='o', color=colors[0])
    Medium = plt.scatter(Medium_Returns['Returns'], Medium_Returns['Variances'], marker='o', color=colors[1])
    Acceptable  = plt.scatter(Acceptable_Returns['Returns'], Acceptable_Returns['Variances'], marker='o', color=colors[2])
    Close0  = plt.scatter(Closezero['Returns'], Closezero['Variances'], marker='o', color=colors[3])
    Negative  = plt.scatter(Negative_Returns['Returns'], Negative_Returns['Variances'], marker='o', color=colors[4])

#Adding features to the plot
    plt.title('K-means plot')
    plt.xlabel('Returns')
    plt.ylabel('Variances')
    plt.legend((Good, Medium, Acceptable, Close0, Negative),
           ('Really good Positive Returns','Medium Returns', 'Acceptable Positive Returns','Close to zero Returns','Negative Returns'),
           loc='upper right')
    plot = plt.savefig('K-means Plot')
    plot = plt_to_np(plot)
    return plot


#Create a diversify portfolio with only the two best performances on each cluster
def diversed_port(data):

    symbol=[]
    for i in range (0,5):
        a = data[data['Cluster_labels']==i].sort_values(by="Returns",ascending=False)
        a = a[a['Cluster_labels']==i]['Stock Symbols'][:2].tolist()
        symbol.append(a)
        single_list = reduce(lambda x,y: x+y, symbol)
    return single_list


#Create a portfolio given one of the five legends, in this fucntion we use as default 'Really good returns'
def portfolios(data ,leg = 'Really good Positive Returns' ):

# The legends are  'Really good Positive Returns','Medium Returns', 'Acceptable Positive Returns','Close to zero Returns','Negative Returns'
    a = data.loc[(data['Cluster_legend'] == leg )].sort_values(by="Returns",ascending=False)
    a = a['Stock Symbols'].tolist()
    return a

def getdata_momentum(tickers,start,end):

#load the data in one DataFrame
    ind_data = pd.DataFrame()
    for t in tickers:
        ind_data[t] = pdr.DataReader(t,data_source='yahoo', start= start, end =end )['Adj Close']
    

#calculate the monthly returns and variance
    mtl_ret = ind_data.pct_change().resample('M').agg(lambda x:(x+1).prod()-1)

#calculate returns over the past 11 months
    past_11 = (mtl_ret+1).rolling(11).apply(np.prod)-1
    index_list = [i for i, item in enumerate(past_11.index)]
    reference_dict = dict(zip(index_list, past_11.index))
    for key in reference_dict.keys():
        key, reference_dict[key]
    """In this part we can see the returns during the study time period. """
    formation = reference_dict[len(reference_dict)-2]
    

#loop through reference_dict, match with formation
    for key in reference_dict.keys():
        if formation == reference_dict[key]:
            previous_month = reference_dict[key-1]
    index_list = [i for i, item in enumerate(past_11.index)]
    reference_dict = dict(zip(index_list, past_11.index))

    for key in reference_dict.keys():
        key, reference_dict[key]
    ret_12 = past_11.loc[previous_month]  
    ret_12 = ret_12.reset_index()


# Get the returns of each quintil
    ret_12['quintile'] = pd.qcut(ret_12.iloc[:,1],5,labels=False)
    return ret_12  

#Get the Database wuth the respective quintiles 
def get_quintiles(data, quintil):

#With this Function the user will be able to identitify the different quintiles in the model just need to assign the number 
  group = data[data.quintile==quintil]['index'].sort_values(ascending=False).tolist()[:5]
  return group

#test the assets function
if __name__=='__main__':
  market ='DJI'
  tickers = assets(market)
  print(tickers)    
    

#test the getdata_means function
if __name__=='__main__':
    tickers
#get prices for the dji for the last year
    start = dt.datetime(2021,3,31)
    end = dt.datetime(2022,3,31)
    dataframe  = getdata_kmeans(tickers,start,end)
    print(dataframe)


##test the elbow_method function
if __name__=='__main__':
    data = pd.read_csv('Sample_Data/sample1.csv')
    result = elbow_method(data)
    print(result)

#test the bestteachclust s function
if __name__=='__main__':
    data1 = pd.read_csv('Sample_Data/sample2.csv')
    result = besteachclust(data1)
    print(result)
    
##test the plotall function
if __name__=='__main__':
    data1 = pd.read_csv('Sample_Data/sample3.csv')
    result = plotall(data1) 
    print(result)

#test the assetsdiversed_port function
if __name__=='__main__':
    data1 = pd.read_csv('Sample_Data/sample2.csv')
    result = diversed_port(data1) 
    print(result)

#test the portfolios function
if __name__=='__main__':
    data1 = pd.read_csv('Sample_Data/sample3.csv')
    result = portfolios(data1, 'Really good Positive Returns') 
    print(result)

#test the getdata_momentum function    
if __name__=='__main__':
    tickers
    start = dt.datetime(2021,3,31)
    end = dt.datetime(2022,3,31)
    dataframe  = getdata_momentum(tickers,start,end)
    dataframe.to_csv('Sample_Data/samplemomentum.csv' )
    print(dataframe)

#test the get_quintiles function
if __name__=='__main__':
    data = pd.read_csv('Sample_Data/samplemomentum.csv')
    quintil =1
    lista = get_quintiles(data, quintil)
    print(lista)

