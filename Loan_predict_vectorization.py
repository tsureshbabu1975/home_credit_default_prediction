#import packages to do vectorization
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.sparse import coo_matrix,hstack,vstack
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder
import pickle

class vectorization:
    #Below method will transform all categorical values to 
    def convert_cat_to_vector(self,cat_features):
        cat_feature_names=[]
        with open('./Data/cat_encoder_fit.pkl', 'rb') as f:
            oh_enc = pickle.load(f)
        tr_cat_vector = oh_enc.transform(cat_features)
        cat_feature_names =list(oh_enc.get_feature_names())
        return tr_cat_vector,cat_feature_names
    
    def convert_num_to_vector(self,num_columns,num_features):
        #load already trained numerical transformer
        with open('./Data/num_encoder_fit.pkl', 'rb') as f:
            scaler = pickle.load(f)
        tr_num_vector =scaler.transform(num_features)
        return  tr_num_vector,  list(num_columns) 

class Home_credit_default_vectorization:
    def vectorize_train_data():
        #read preprocessed train data
        df = pd.read_csv("./Data/pre_processed_df_train.csv",sep='\t', encoding='utf-8')
        #any -inf, nan value fill with 0 
        #df.replace([np.inf, -np.inf,np.NaN,np.nan], 0, inplace=True)
        y = df['TARGET']
        #cat columns 
        cat_cols=df.select_dtypes(include=object).columns.to_list()
        #numerical columns
        num_cols=df.select_dtypes(exclude=object).columns.to_list()
         #any inf, nan value filled with zero
        df.replace([np.inf, -np.inf,np.NaN,np.nan],0, inplace=True)           
        #split train and test
        #split dataset rows into 67 and 33% 
        train_slice =int(0.67*(len(df)))
        X_train =pd.DataFrame(np.array(df[:train_slice]),columns=df.columns)
        X_test =pd.DataFrame(np.array(df[train_slice:]),columns=df.columns)
        X_train =pd.DataFrame(X_train,columns=df.columns)
        X_test =pd.DataFrame(X_test,columns=df.columns)               
        #since dataset is split into train and test, delete dataset to reduce memory
        gc.enable()
        del df
        gc.collect()
        #invoke vectorization categorical convertion method to convert all categorical features
        obj_vect =vectorization()
        #invoke categorical response coding method to convert categorical values to numerical vector
        #for training, invoke train and test separately 
        tr_cat_vect,cat_feature_names= obj_vect.convert_cat_to_vector(X_train[cat_cols])
        te_cat_vect,cat_feature_names= obj_vect.convert_cat_to_vector(X_test[cat_cols])
        #remove unnecessary columns
        num_cols.remove('Unnamed: 0')
        num_cols.remove('SK_ID_CURR')
        num_cols.remove('1')
        num_cols.remove('TARGET')
        #convert numerical fetures to vector form using normalizer
        tr_num_vect,num_feature_names= obj_vect.convert_num_to_vector(num_cols,X_train[num_cols])
        te_num_vect,num_feature_names= obj_vect.convert_num_to_vector(num_cols,X_test[num_cols])
        #### since dataset is huge, try to delete /carbage it manually to improve the performance
        gc.enable()
        del X_train,X_test
        gc.collect()
        #merge categorical and numerical vectorized data and feature names.
        feature_names=num_feature_names+cat_feature_names
        X_train_vector = hstack((coo_matrix(tr_num_vect),tr_cat_vect))
        X_test_vector = hstack((coo_matrix(te_num_vect),te_cat_vect))
        #merge all train records
        X_vector_final = vstack((X_train_vector,X_test_vector))
        
        gc.enable()
        del X_train_vector,X_test_vector
        gc.collect()
        #convert final train vector to dataframe
        df_vector_final = pd.DataFrame(X_vector_final.toarray(), columns =feature_names)
        #add class labels back to vectorized dataset
        df_vector_final['TARGET']=y
        #store vectorized df as pickle file
        with open('./Data/df_train_vectorized.pkl', 'wb') as f:
            pickle.dump(df_vector_final, f)
        with open('./Data/train_vectorized_features.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        #print the shapes
        print("Train Final Data matrix ...")
        print("="*25)
        print(X_vector_final.shape)
        print(len(feature_names))
        #return feature names to compare train and test features are same or different
        return feature_names
    
    def vectorize_test_data(df):
        #read preprocessed train data
        #df = pd.read_csv("./Data/pre_processed_df_test.csv",sep='\t', encoding='utf-8')
        #any inf, nan value filled with zero
        #df.replace([np.inf, -np.inf,np.NaN,np.nan], 0, inplace=True)
        id = df['SK_ID_CURR']
        #cat columns 
        cat_cols=df.select_dtypes(include=object).columns.to_list()
        #numerical columns
        num_cols=df.select_dtypes(exclude=object).columns.to_list() 
        #any inf, nan value filled with zero   
        df.replace([np.inf,-np.inf,np.NaN,np.nan],0, inplace=True)   
        #invoke vectorization categorical convertion method to convert all categorical features
        obj_vect =vectorization()
        #invoke categorical response coding method to convert categorical values to numerical vector
        cat_vect,cat_feature_names= obj_vect.convert_cat_to_vector(np.array(df[cat_cols]))
        #remove unnecessary columns
        #num_cols.remove('Unnamed: 0')
        num_cols.remove('SK_ID_CURR')
        num_cols.remove('1')
        #convert numerical fetures to vector form using normalizer
        num_vect,num_feature_names= obj_vect.convert_num_to_vector(num_cols,np.array(df[num_cols]))
        #### since dataset is huge, try to delete /carbage it manually to improve the performance
        gc.enable()
        del df
        gc.collect()
        #merge categorical and numerical vectorized data and feature names.
        feature_names=num_feature_names+cat_feature_names 
        X_vector_final = hstack((coo_matrix(num_vect),cat_vect))
        #convert final train vector to dataframe
        df_vector_final = pd.DataFrame(X_vector_final.toarray(), columns =feature_names)
        #add id column back to dataframe to upload for kaggle score
        df_vector_final['SK_ID_CURR']=id
        #store vectorized df as pickle file
        with open('./Data/df_test_vectorized_streamlit.pkl', 'wb') as f:
            pickle.dump(df_vector_final, f)
        with open('./Data/test_vectorized_features.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        #print the shapes
        #print("Final Data matrix ...")
        #print("="*25)
        #print(X_vector_final.shape)
        #print(len(feature_names))
        #return feature names to compare train and test features are same or different
        return df_vector_final
    
#train_features = Home_credit_default_vectorization.vectorize_train_data()
#df = pd.read_csv("./Data/pre_processed_df_test.csv",sep='\t', encoding='utf-8')
#test_features = Home_credit_default_vectorization.vectorize_test_data(df)
#print('Difference of train and test columns ~~>{}'.format(set(train_features).difference(test_features)))

