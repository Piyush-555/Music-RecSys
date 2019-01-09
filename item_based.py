import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import gc

class item_based_recommender():
    def __init__(self, songs_data, train_data, test_data, n_users, n_items):
        self.songs_data=songs_data
        self.train_data=train_data
        self.test_data=test_data
        self.n_users=n_users
        self.n_items=n_items
        self.similarity_matrix=None

    def create_user_item_matrix(self, data, is_sparse, dtype):
        ui_matrix=np.zeros((self.n_items, self.n_users),dtype=dtype)
        for index,row in data.iterrows():
            ui_matrix[row['song_idx']-1,row['user_idx']-1]=row['play_count']
        if is_sparse==True:
            ui_matrix=csr_matrix(ui_matrix, dtype=dtype)
        return ui_matrix

    def create_similarity_matrix(self, ui_matrix, dtype):
        similarity_matrix=cosine_similarity(ui_matrix).astype(dtype)
        self.similarity_matrix=similarity_matrix
        return similarity_matrix

    def get_users_songs(self, user_idx):
        users_songs=self.train_data.loc[self.train_data['user_idx']==user_idx]
        return users_songs

    def recommend(self, ui_matrix, user_idx, number_of_recommendations=10):
        users_songs=self.get_users_songs(user_idx)['song_idx'].tolist()
        scores=self.similarity_matrix@ui_matrix[:,user_idx-1]
        print(scores.shape)
        similarity_sum=self.similarity_matrix.sum(axis=1)
        print(similarity_sum.shape)
        similarity_sum=similarity_sum.reshape(self.n_items,1)
        print(similarity_sum.shape)
        scores=np.divide(scores, similarity_sum)
        print(scores.shape)
        scores=np.c_[scores, range(1,self.n_items+1)]
        scores=scores.tolist()
        scores.sort(key=itemgetter(0), reverse=True)
        recommendations=pd.DataFrame(data=scores,columns=['score','song_idx'])
        recommendations=recommendations.merge(self.songs_data,on='song_idx')
        return recommendations.head(number_of_recommendations)
            
        

if __name__=="__main__":
    org_df=pd.read_csv('Dataset/user_data.csv')
    org_df=org_df.drop('Unnamed: 0', axis=1)

    main_df=org_df.head(100000)
    n_users=len(main_df.user_id.unique())
    n_songs=len(main_df.song_id.unique())
    print("Users: ",n_users)
    print("Songs: ",n_songs)

    user_id_df=main_df[['user_id']]
    user_id_df['songs_listened']=1
    user_id_df=user_id_df.groupby(['user_id']).agg({'songs_listened':'count'}).reset_index()
    user_id_df['user_idx']=range(1,n_users+1)

    song_id_df=main_df[['song_id']]
    song_id_df['users_listened']=1
    song_id_df=song_id_df.groupby(['song_id']).agg({'users_listened':'count'}).reset_index()
    song_id_df['song_idx']=range(1,n_songs+1)

    main_df=main_df.merge(user_id_df,on='user_id')
    main_df=main_df.merge(song_id_df,on='song_id')

    data=main_df[['user_idx','song_idx','play_count','title','artist_name']]
    train_data,test_data=train_test_split(data,test_size=0.2)
    songs_data=data[['song_idx','title','artist_name']].drop_duplicates('song_idx')

    ibcf=item_based_recommender(songs_data=songs_data, train_data=train_data, test_data=test_data, n_users=n_users, n_items=n_songs)
    del data, main_df, org_df, user_id_df, song_id_df, songs_data
    gc.collect()

    ui_train=ibcf.create_user_item_matrix(data=train_data, is_sparse=True, dtype=np.int16)
    ui_test=ibcf.create_user_item_matrix(data=test_data, is_sparse=True, dtype=np.int16)
    item_item_similarity_matrix=ibcf.create_similarity_matrix(ui_train, dtype=np.float16)
    
