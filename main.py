import os 
os.system('python data_exploration.py')

 # some therotical words before starting

 #1) content-based recommender systems focus on the properties of the items and give you recommendations based on the similarity between them.

# 2) Collaborative filtering produces recommendations based on the knowledge of users’ preference to items

"""
a- Memory-Based Collaborative Filtering
  1- Item-Item Collaborative Filtering: “Users who liked this item also liked …
   2- User-Item Collaborative Filtering: “Users who are similar to you (kinda like the twin you never knew you had) also liked …


b- Model-Based Collaborative Filtering
   Model-based CF methods are based on matrix factorization 
   Matrix factorization restructures the user-item matrix into a low-rank matrix
"""

import pandas as pd 
import numpy as np 
data=pd.read_csv('ml-100k/u.data',sep='\t')
data.columns=['user id','item id', 'rating', 'timestamp']
#print(data.head())


# preprocessing
data.drop('timestamp',axis='columns', inplace=True)
print('data sample ')
print(data.head())

# count of nulls
print("number of nulls", data.isna().sum())


ratings_df = data.pivot(
    index='user id',
    columns='item id',
    values='rating'
)

ratings = ratings_df.fillna(0).values
print('new data ', ratings)

#------------------------------------------------
# split into training and testing
delete_rating_count = 2
min_user_ratings = 30
def train_test_split(ratings):
    validation = np.zeros(ratings.shape)
    train = ratings.copy()

    for user in np.arange(ratings.shape[0]):
        if len(ratings[user,:].nonzero()[0]) >= min_user_ratings:
            np.random.seed(0)
            val_ratings = np.random.choice(
              ratings[user, :].nonzero()[0],
              size=delete_rating_count,
              replace=False
            )
            train[user, val_ratings] = 0
            validation[user, val_ratings] = ratings[user, val_ratings]
    return train, validation
train, test =train_test_split(ratings)
#print(train.shape,test.shape)
#We remove some existing ratings from users by replacing them with zeros.
#----------------------------------------------------
# svd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train, k = 20)#tweak k, dimensionality for rank matrix
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

from losses import rmse_loss

# predict on the validation dataset
#print ('matrix-factorization CF RMSE: %.2f' #%rmse_loss.rmse(X_pred, test))

#_____________________________________________
#prediction function 
"""The matrix factorization approach reduces the dimensions of the rating matrix r by factorizing it into a product of two latent factor matrices, p for the users and q for movies"""
#P is latent user feature matrix
#Q is latent movie feature matrix
def prediction(P,Q):
    return np.dot(P.T,Q)

#---------------------------------------------
# Stochastic gradient descent Method 
# parameters of optomization 
lmbda = 0.4 # Regularization parameter
k = 3 #might be changed 
m, n = train.shape  # Number of users and movies
n_epochs = 200 # Number of epochs / (max number of epochs)
alpha=0.0075  # Learning rate
#print('m',m,'n',n)

# first intialization method
np.random.seed(0)
P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent movie feature matrix

#second intialization method 
""""
import math
average_rating= data["rating"].mean()
const = math.sqrt (average_rating/k)
P= np.full((k, m),const)
Q= np.full((k, n),const) """

#print("p",P)
#----------------------------------------------

#optimization / training  process 
train_errors = []
val_errors = []

#---------------------------------------------------------

# SGD 
#Psudocode
#1-Initialize user factor and movie factor matrices
#2-For each epoch and for each batch, run gradient descent to update the user factor matrix while fixing the item factor matrix. Then run gradient descent to update the item factor matrix using the updated user factor matrix.
#3-Repeat until converged.
#for i in range(num_epochs):
   # np.random.shuffle(data)
    #for example in data:
       # grad = compute_gradient(example, params)
        #params = params — learning_rate * grad
tol= 10-9
epoch=0
#Only consider items with ratings 
users,items = train.nonzero()    
  
while epoch < (n_epochs):
    # keep track of previous step
    P1=P.copy()
    Q1=Q.copy()
    for u, i in zip(users,items):
        e = train[u, i] - prediction(P[:,u],Q[:,i])  
        #print('error',e)
        # Calculate error for gradient update
        P[:,u] += alpha * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
        Q[:,i] += alpha * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix
    train_rmse = rmse_loss.rmse(prediction(P,Q),train)
    val_rmse = rmse_loss.rmse(prediction(P,Q),test) 
    train_errors.append(train_rmse)
    val_errors.append(val_rmse)
    if (np.linalg.norm(P1-P) <=tol and np.linalg.norm(Q1-Q)    <=tol):
      break
    if(epoch%5==0):
      print(epoch ,train_rmse,val_rmse)
    
    epoch = epoch+1 
    
#---------------------------------------------

# predicted prediction vs actual ones
SGD_prediction=prediction(P,Q)
estimation= SGD_prediction[test.nonzero()]
ground_truth = test[test.nonzero()]
results=pd.DataFrame({'prediction':estimation, 'actual rating':ground_truth})
results.to_csv('Results/flex_epoch_lr_0.0075_K_'+str(k)+'_results_rmse_loss_stochastic_gradient_descent.csv')
print(results.head())
pd.DataFrame(train_errors).to_csv('Results/flex_epoch_lr_0.0075_K_'+str(k)+'_train_errors_rmse_loss_stochastic_gradient_descent.csv')
pd.DataFrame(val_errors).to_csv('Results/flex_epoch_lr_0.0075_K_'+str(k)+'_val_errors_rmse_loss_stochastic_gradient_descent.csv')

#--------------------------------------------------

#evaluation 

# reading results
df_results =pd.read_csv('Results/flex_epoch_lr_0.0075_K_'+str(k)+'_results_rmse_loss_stochastic_gradient_descent.csv')

# creating random results 
import random
random.seed(0)
rand= [random.uniform(1, 5) for i in range(0, len(df_results)  )]

# calc RMSE for random guess 
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_rand = sqrt(mean_squared_error(df_results['actual rating'], rand))

# Calculating RMSE with predictions on test set 
rmse_model = sqrt(mean_squared_error(df_results['actual rating'], df_results['prediction']))

print ("RMSE of SGD on test set",rmse_model)
print("RMSE of random guess on test set",rmse_rand)
print("Required number of epochs",epoch)

