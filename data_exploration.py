 # Exploring the data 

import pandas as pd

#user id | item id | rating | timestamp.

data=pd.read_csv('ml-100k/u.data',sep='\t')
data.columns=['user id','item id', 'rating', 'timestamp']
print(data.head())
print(data.shape)


# genre_decoding
genre_decoding=pd.read_csv('ml-100k/u.genre',sep='|')
genre_decoding.columns=['genre','code']
print(genre_decoding)



# print info  about items     
#               movie id | movie title | release date |video release date |
             # IMDb URL | unknown | Action | Adventure | Animation |
             # Children's | Comedy | Crime | Documentary | Drama | Fantasy |
             # Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              #Thriller | War | Western |
print('items info')
items_info=pd.read_csv('ml-100k/u.item',sep='|',encoding='latin-1')

items_info.columns=[ 'movie id', 'movie title', 'release date', 'video release date',
              'IMDb URL',  'unknown' ,'Action' , 'Adventure' ,'Animation',
              'Children', 'Comedy' ,'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi',
              'Thriller','War' ,'Western']
print(items_info.head())
print(items_info.shape)

# avarage rating 

print('Average rating is:', data["rating"].mean())

#import matplotlib.pyplot as plt


# Top rated movies
#fig, ax = plt.subplots(figsize=(5,5))
print('Top rated movies')
grouped_data=data.groupby(['item id']).sum()['rating']
grouped_data=grouped_data.reset_index()
print(grouped_data.sort_values(by='rating', ascending=False))

