from pickletools import optimize
from pandas_datareader import Options
import mean
import datetime as dt

#Define User input
userinput = {
        "start" : dt.datetime(2021,1,1),
        "end" : dt.datetime(2022,4,4),
        "market": "DJI",
        "optimization": "Acceptable Positive Returns"
    }

#Run the model 
def main(market, start, end, optimization):
    data = mean.getdata_kmeans(market,start=start,end=end)

    #Run the model elbow method
    elbow_plot, data = mean.elbow_method(data)
    image1 = mean.plt_to_np(elbow_plot)



    #Define the best clusters
    data = mean.besteachclust(data)

    #Plot all the assets 
    K_meansplot = mean.plotall(data)
    image2 = mean.plt_to_np(K_meansplot)

    #Diversified portfolio
    Diversified_portfolio = mean.diversed_port(data)

    #Best Portfolio possible Options:
    #Really good Positive Returns','Medium Returns', 'Acceptable Positive Returns','Close to zero Returns','Negative Returns'
    Result_kmeans =mean.portfolios(data,leg=optimization)
   
    #Define User Outputs
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


