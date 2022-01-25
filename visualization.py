# importing package
import matplotlib.pyplot as plt
import pandas as pd 

df2= pd.read_csv('Results/train_errors_rmse_loss_stochastic_gradient_descent.csv')
df3= pd.read_csv('Results/val_errors_rmse_loss_stochastic_gradient_descent.csv')
# create data
x = df2['Unnamed: 0']

  
# plot lines
plt.plot(x, df2['0'], label = "train loss")
plt.plot(x, df3['0'], label = "test loss")
plt.title('Monitoring loss during SGD training')
plt.legend()
plt.savefig('Results/Monitoring_loss_duting_SGD_training.png')
plt.show()
