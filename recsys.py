# music recsys

# Dataset used == Million Song Dataset

# Importing Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Models import *
import sqlite3


#Loading signedup users (Demo)
query=User.select(User.user_id,User.password)
user_info_df=pd.DataFrame(list(query.dicts()))

'''
# Loading Data
users_df=pd.read_csv('Dataset/train_triplets.txt',sep='\t',header=None)
users_df.columns=['user_id','song_id','play_count']


conn=sqlite3.connect('Dataset/track_metadata.db')
meta_df=pd.read_sql_query('SELECT * FROM songs',conn)
conn.close()
'''['track_id',
 'title',
 'song_id',
 'release',
 'artist_id',
 'artist_mbid',
 'artist_name',
 'duration',
 'artist_familiarity',
 'artist_hotttnesss',
 'year',
 'track_7digitalid',
 'shs_perf',
 'shs_work']'''

genre_df=pd.read_csv('Dataset/MAGD_generes.cls',sep='\t',header=None)
genre_df.columns=['track_id','genre']


main_df=users_df.merge(meta_df.drop_duplicates(['song_id']),how='left',on='song_id')
    
popularity_df=users_df[['song_id','play_count']] 
popularity_df=popularity_df.groupby(['song_id']).agg({'play_count': 'count'}).reset_index()
popularity_df=popularity_df.merge(meta_df[['track_id','song_id','title','artist_name','artist_familiarity']].drop_duplicates(['song_id']),on="song_id",how="left")  
popularity_df=popularity_df[popularity_df.play_count>=50].sort_values(by=['play_count'],ascending=False)
popularity_df=popularity_df.merge(genre_df,on='track_id',how='left')

    
users_combined=popularity_df[['song_id','title','artist_name']]
users_combined['song_popularity']=range(1,len(users_combined)+1)    
users_combined=users_df.merge(users_combined,on="song_id",how="left") 
users_combined=users_combined.dropna()   
users_combined=users_combined.sort_values(by=['user_id','song_popularity'],ascending=[True,True])  
    
    
test=users_combined.head(1000000)
print("Users:", len(test.user_id.unique())) 
print("Songs:", len(test.song_id.unique()))   
print("Artists:", len(test.artist_name.unique()))     

'''

'''    
#Saving datasets
popularity_df.to_csv('Dataset/popularity_based.csv',index=True)
popularity_df.dropna().to_csv('Dataset/popularity_genre_based.csv',index=True)
users_combined.to_csv('Dataset/user_data_full.csv',index=True)
test.to_csv('Dataset/user_data.csv',index=True)
'''    
    
#cosine similarity between users of song i and j 
#(i,j belongs to songs from playlist of user to whom songs are being recommended)

# Popularity based recommender using self modified data
# Modification of data is commented out
# Data is modified from full million_song_dataset_metadata, echonest_taste_profile_data and Top_MAGD_dataset           
class popularity_recommender():
    def __init__(self):
        self.df=pd.read_csv('Dataset/popularity_based.csv')
        self.df=self.df.drop('Unnamed: 0',axis=1)
        
        
    def normalize_columns(self,list_of_col):
        for feature in list_of_col:
            self.df[feature]=(self.df[feature]-self.df[feature].min())/(self.df[feature].max()-self.df[feature].min())    
        
    def get_columns(self):
        columns=list(self.df.columns)
        return columns
    
    def get_unique_genres(self,listed=False):
        genres=list(self.df.genre.unique())
        idx=[i for i in genres if type(i)==float]
        genres.remove(idx[0])
        if listed==True:
            return genres
        return len(genres)
    
    def get_unique_songs(self):
        songs=list(self.df.song_id.unique())        
        return len(songs)
    
    def get_unique_artists(self):
        artists=list(self.df.artist_name.unique())        
        return len(artists)
        
    def recommend(self,number_of_recommendations=10,artist_list=None,genre_list=None):
        
        if artist_list!=None and genre_list!=None:
            genre_list.append('nan')
            self.df=self.df.fillna('nan')
            count=0
            for index, row in self.df.iterrows():
                if row['genre'] in genre_list and row['artist_name'] in artist_list:
                    count+=1
                    print(row['title'],"-",row['artist_name'])
                if count==number_of_recommendations:
                    break
                
        elif artist_list!=None:
            count=0
            for index, row in self.df.iterrows():
                if row['artist_name'] in artist_list:
                    count+=1
                    print(row['title'],"-",row['artist_name'])
                if count==number_of_recommendations:
                    break 
                
        elif genre_list!=None:
            count=0
            for index, row in self.df.iterrows():
                if row['genre'] in genre_list:
                    count+=1
                    print(row['title'],"-",row['artist_name'])
                if count==number_of_recommendations:
                    break
        else:
            count=0
            for index, row in self.df.iterrows():
                count+=1
                print(row['title'],"-",row['artist_name'])
                if count==number_of_recommendations:
                    break
                
########################################################
'''
users_full=pd.read_csv('Dataset/user_data_full.csv')
users_full=users_full.drop('Unnamed: 0',axis=1)
users_random=users_full.sample(1000000,random_state=0)
ratings=users_random['play_count'].values
v1=(ratings-ratings.mean())/ratings.std()
from matplotlib import pyplot as plt
plt.hist(v1,bins=1000)
plt.show()
'''
users_df=pd.read_csv('Dataset/user_data.csv')
users_df=users_df.drop('Unnamed: 0',axis=1)
#22798 unique users
#90328 unique songs

users_df=users_df.head(100000)
print("Users",len(users_df.user_id.unique()))
print("Songs",len(users_df.song_id.unique()))
#Users 2235
#Songs 38642
'''
['user_id', 'song_id', 'play_count', 'title',
       'artist_name', 'song_popularity']
'''
user_id_df=users_df[['user_id']]
user_id_df['songs_listened']=1
user_id_df=user_id_df.groupby(['user_id']).agg({'songs_listened':'count'}).reset_index()
user_id_df['songs_listened'].values.mean()
#avg songs listened==44.4049     
user_id_df['user_idx']=range(1,2236)  
users_df=users_df.merge(user_id_df,on='user_id')


song_id_df=users_df[['song_id']]
song_id_df['users_listened']=1
song_id_df=song_id_df.groupby(['song_id']).agg({'users_listened':'count'}).reset_index()       
song_id_df['song_idx']=range(1,38643)  
users_df=users_df.merge(song_id_df,on='song_id')

data=users_df[['user_idx','song_idx','play_count','title','artist_name']]
from sklearn.cross_validation import train_test_split
train_data,test_data=train_test_split(data,test_size=0.2)
print(train_data.shape)

#User-Item Matrix
#Songs are rows and Users are columns
ui_train=np.zeros((38642, 2235),dtype=np.int16)
for index,row in train_data.iterrows():
    ui_train[row['song_idx']-1,row['user_idx']-1]=row['play_count']

ui_test=np.zeros((38642, 2235),dtype=np.int16)    
for index,row in test_data.iterrows():
    ui_test[row['song_idx']-1,row['user_idx']-1]=row['play_count']    

from scipy.sparse import csr_matrix
ui_train=csr_matrix(ui_train,dtype=np.int16)

#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
item_sim=cosine_similarity(ui_train).astype(np.float16)

# my happy ending 4638
i_4638=item_sim[:,4637]
result=item_sim@i_4638
recommend=[]
for i in range(38642):
    recommend.append((i+1,result[i]))

from operator import itemgetter
recommend.sort(key=itemgetter(1))
recommend.reverse()


song_id_df=song_id_df.merge(train_data[['song_idx','title','artist_name']].drop_duplicates('song_idx'),on='song_idx')

for i in recommend:
    song_idx=i[0]
    for index,row in song_id_df.iterrows():
        if song_idx==row['song_idx']:
            print(row['title'],"-",row['artist_name'],i[1])
    

########################################################                
                
                
if __name__=="__main__":                
    pr=popularity_recommender()
    pr.normalize_columns(['play_count','artist_familiarity'])
    pr.df['score']=2*pr.df['play_count']+3*pr.df['artist_familiarity']        
    pr.df=pr.df.sort_values(by='score',ascending=False)
    
    pr.recommend(20,genre_list=['Pop_Rock'])
        