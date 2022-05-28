from pandas_datareader import Options
import mean
import A_Kmeans
import B_Momentum
import datetime as dt

userinput = {
        "start" : dt.datetime(2021,1,1),
        "end" : dt.datetime(2022,4,4),
        'Market': 'DJI'
        "Model":'K-means',
        'quintil or optimization': 'Medium Returns'
        
    }
#Define the portfolio
# The user can decided between 'Kmeans and Mommetum'
tick = test2.portfopt('Momentum')



# Define the time to evaluate   
start = dt.datetime(2021,1,1)
end = dt.datetime(2022,4,4)
data = test2.getdata_opt_port(tick,start=start,end=end)

# Define the neccesary functions to run the model and obtain the optimization portfolio

'''a = test1.cov_matrix(data, index_col='Date')
b = test1.corr_matrix(data, index_col='Date')
c= test1.weights (data, index_col='Date')

e = test1.port_returns(data)
d = test1.port_ind_exp(data)'''
f = test2.total_opt (data)
print(f[0])
print(f[1])

outputs = {

         "Diversified Port " :  Diversified_portfolio ,
         "optimization port": Result_kmeans,
         "elbow plot":image1,
         "K_means Plot": image2
    

     }

    return print(outputs)




if __name__=='__main__':
    #Define the market 
    # Options 'DJI', 'S&P500' and 'ASX'
    market = mean.assets (userinput["market"])
    # Define the time to evaluate   
    start = userinput['start']
    end = userinput['end']
    optimization = userinput["optimization"]
    #this is the starting point
    #main is the app / service

  

   


    

    Result_kmeans = main(market, start , end, optimization)


