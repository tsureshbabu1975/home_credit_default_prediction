#import packages to do vectorization
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix,hstack
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

class vectorization:
    #Below method will transform all categorical values to 
    def convert_cat_to_vector(self,cat_features):
        cat_feature_names=[]
        with open('./data/cat_encoder_fit.pkl', 'rb') as f:
            oh_enc = pickle.load(f)
        tr_cat_vector = oh_enc.transform(cat_features)
        cat_feature_names =list(oh_enc.get_feature_names())
        return tr_cat_vector,cat_feature_names
    
    def max_length(self,lines):
        return max([len(s.split()) for s in lines])
    def compute_class_weights(self,y):
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0,1], y=y)
        return dict(enumerate(class_weights))

class Home_credit_default_vectorization:
    def vectorize_train_data():
        #read preprocessed train data
        df = pd.read_csv("./data/pre_processed_df_train.csv",sep='\t', encoding='utf-8')
        #any -inf, nan value fill with 0 
        #df.replace([np.inf, -np.inf,np.NaN,np.nan], 0, inplace=True)
        y = df['TARGET']
        #remove unnecessary columns
        df.drop(columns=['Unnamed: 0','SK_ID_CURR','1','TARGET'],axis=1,inplace=True)
         #any inf, nan value filled with zero
        df.replace([np.inf, -np.inf,np.NaN,np.nan],0, inplace=True)           
        #cat columns 
        cat_cols=df.select_dtypes(include=object).columns.to_list()
        #num columns
        num_cols=df.select_dtypes(exclude=object).columns.to_list()
        # dimensions of each categorical feature
        cat_dimensions = df[cat_cols].nunique().tolist()
        #invoke vectorization categorical convertion method to convert all categorical features
        obj_vect =vectorization()
        #compute class weights as it is highly imbalanced dataset. 
        #Hence class weights are required at the time of training the model
        d_cls_wghts =obj_vect.compute_class_weights(y)
        #compute categorical embeddings
        cat_embedding_dim=[obj_vect.max_length(df[cat]) for cat in cat_cols]        
        #invoke categorical method to convert categorical values to numerical
        cat_vect,cat_feature_names= obj_vect.convert_cat_to_vector(df[cat_cols])
        #for tabnet keep the numerical features as is, and doesn't require Feature Tranforma
        #merge categorical and numerical vectorized data and feature names.
        feature_names=num_cols+cat_feature_names
        #final vector passed on to the model
        X_vector_final = hstack((np.array(df[num_cols]),coo_matrix(cat_vect)))
        gc.enable()
        del cat_vect
        gc.collect()
        #convert final train vector to dataframe
        df_vector_final = pd.DataFrame(X_vector_final.toarray(), columns =feature_names)
        #compute categorical features indexes in the feature array. TabNet explicitly requires
        cat_indexes = [column_index for column_index, column_name in enumerate(df_vector_final.columns) if column_name in cat_cols]
        #convert y class values to label encoding
        slice=int(0.67*len(y))
        label_encoder = LabelEncoder()
        label_encoder.fit(y[:slice]) 
        df_vector_final['TARGET']=label_encoder.transform(y)   
        
        #store categorical indexes, categorical dimensions, categorical embedding dimensions in a dictionary
        d_utilities = {} #dict
        d_utilities['cat_dimensions']=cat_dimensions
        d_utilities['cat_indexes']=cat_indexes
        d_utilities['cat_embedding_dim']=cat_embedding_dim
        d_utilities['d_cls_wghts']=d_cls_wghts
        #store utilitiy dictionary  as pickle file
        with open('./data/utilities_file.pkl', 'wb') as f:
            pickle.dump(d_utilities, f)
        #store vectorized df as pickle file
        with open('./data/df_train_vectorized_dl.pkl', 'wb') as f:
            pickle.dump(df_vector_final, f)
        #print the shapes
        print("Train Final Data matrix ...")
        print("="*25)
        print(X_vector_final.shape)
        print(len(feature_names))
        #return feature names to compare train and test features are same or different
        return feature_names
    
    def vectorize_test_data():
        #read preprocessed train data
        df = pd.read_csv("./data/pre_processed_df_test.csv",sep='\t', encoding='utf-8')
        #any inf, nan value filled with zero
        #df.replace([np.inf, -np.inf,np.NaN,np.nan], 0, inplace=True)
        id = df['SK_ID_CURR']
        #remove unnecessary columns
        df.drop(columns=['Unnamed: 0','SK_ID_CURR','1'],axis=1,inplace=True)        
        #cat columns 
        cat_cols=df.select_dtypes(include=object).columns.to_list()
        #numerical columns
        num_cols=df.select_dtypes(exclude=object).columns.to_list() 
        #any inf, nan value filled with zero   
        df.replace([np.inf,-np.inf,np.NaN,np.nan],0, inplace=True)   
        #invoke vectorization categorical convertion method to convert all categorical features
        obj_vect =vectorization()
        #invoke categorical method to convert categorical values to numerical
        cat_vect,cat_feature_names= obj_vect.convert_cat_to_vector(df[cat_cols])
        #for tabnet keep the numerical features as is, and doesn't require Feature Tranforma
        #merge categorical and numerical vectorized data and feature names.
        feature_names=num_cols+cat_feature_names
        X_vector_final = hstack((np.array(df[num_cols]),coo_matrix(cat_vect)))
        gc.enable()
        del cat_vect
        gc.collect()
        #convert final train vector to dataframe
        df_vector_final = pd.DataFrame(X_vector_final.toarray(), columns =feature_names)
        #add id column back to dataframe to upload for kaggle score
        df_vector_final['SK_ID_CURR']=id
        #store vectorized df as pickle file
        with open('./data/df_test_vectorized_dl.pkl', 'wb') as f:
            pickle.dump(df_vector_final, f)
        with open('./data/test_vectorized_features_dl.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        #print the shapes
        print("Test Final Data matrix ...")
        print("="*25)
        print(X_vector_final.shape)
        print(len(feature_names))
        #return feature names to compare train and test features are same or different
        return feature_names
    
train_features = Home_credit_default_vectorization.vectorize_train_data()
test_features = Home_credit_default_vectorization.vectorize_test_data()
print('Difference of train and test columns ~~>{}'.format(set(train_features).difference(test_features)))
print('Difference of test and train columns ~~>{}'.format(set(test_features).difference(train_features)))