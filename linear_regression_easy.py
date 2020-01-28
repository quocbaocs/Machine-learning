import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import math

data = pd.read_csv('linear.csv').values
x = data[:, 0].reshape(-1, 1) # take column 0, all row
y = data[:, 1].reshape(-1, 1) # take colume 1, all row
plt.title("linear regression easy")
plt.scatter(x,y) #plot data 
mean_x = np.mean(x)
mean_y = np.mean(y)

predic_x = x - mean_x
predic_y = y - mean_y

multi = predic_x*predic_y
square_x = predic_x*predic_x

B1 = round(np.sum(multi)/np.sum(square_x),2) # Estimating The Slope

B0 = round(mean_y - (B1*mean_x),1)

predic_y_future = B0 + B1*x # line predict
plt.plot(x,predic_y_future, linewidth=3) #Test 
plt.scatter(x,predic_y_future)
plt.show()

error = predic_y_future - y #calculate the difference between
                            #each model prediction and the actual y values
#We can easily calculate the square of each of these error values (error × error or error 2 ).
square_error = error*error
N = data.shape[0]
print("Estimating Error :",math.sqrt(np.sum(square_error)/N))
