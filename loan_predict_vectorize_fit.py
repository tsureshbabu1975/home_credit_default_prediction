#import packages to do vectorization
from urllib.request import UnknownHandler
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.sparse import coo_matrix,hstack,vstack
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle

class encoder:
    def load_preprocessed_train_data(self):
        #load preprocessed train data
        df =  pd.read_csv('./data/pre_processed_df_train.csv', sep='\t', encoding='utf-8')
        #remove unnecessary columns during preprocessing
        cols =['Unnamed: 0','SK_ID_CURR','1','TARGET']
        df.drop(columns=cols,axis=1,inplace=True)
        return df
    def split_data(self,df):
        #find categorical and numerical fields
        cat_cols=df.select_dtypes(include=object).columns.to_list()
        num_cols=df.select_dtypes(exclude=object).columns.to_list()
        #any inf, nan value filled with zero   
        df.replace([np.inf,-np.inf,np.NaN,np.nan],0, inplace=True)           
        #split categorical data into train and test. 
        #for train will use only train dataset. 
        train_slice =int(0.67*(len(df)))
        X_train_cat =np.array(df[cat_cols][:train_slice].values)
        X_train_num =np.array(df[num_cols][:train_slice].values)
        return X_train_cat, X_train_num
    
    def train_cat_features(self,X_train):
        ohencoder = OneHotEncoder(handle_unknown='ignore')    
        ohencoder.fit(X_train)
        #store vectorized df as pickle file
        with open('./data/cat_encoder_fit.pkl', 'wb') as f:
            pickle.dump(ohencoder, f)        
        return 
    def train_num_features(self,X_train):
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X_train)
        with open('./data/num_encoder_fit.pkl', 'wb') as f:
            pickle.dump(scaler, f)        
        return 
    def train_features():
        obj_enc = encoder()
        df =obj_enc.load_preprocessed_train_data()
        X_cat_features,X_num_features=obj_enc.split_data(df)
        obj_enc.train_cat_features(X_cat_features)
        obj_enc.train_num_features(X_num_features)

encoder.train_features()