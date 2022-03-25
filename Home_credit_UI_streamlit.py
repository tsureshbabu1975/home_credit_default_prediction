import sys 
import os
import pandas as pd
import numpy as np
from pyparsing import col
import streamlit as st
import pickle
#sys.path.append(os.path.abspath("C:/DS/1. AAIC/001.0ML.Project/26.repayloan"))

from Loan_predict_preprocessing import home_credit_default
from Loan_predict_vectorization import Home_credit_default_vectorization
from Loan_predict_LGBM import Home_credit_LGBM

class home_credit_streamlit:
    @st.cache(allow_output_mutation=True)
    def process_application_test(datafile):
        #optimize file size of application train data
        #df_test = pd.read_csv('./data/application_test.csv', encoding= 'unicode_escape', nrows=nrows)
        df_test = home_credit_default.process_data(df=datafile)
        #find categorical and numerical fields
        cat_cols=df_test.select_dtypes(include=object).columns.to_list()

        num_cols=df_test.select_dtypes(exclude=object).columns.to_list()
        #categorical values fill data_not_available        
        df_test[cat_cols].replace([np.inf, -np.inf,np.NaN,np.nan], 'Data_Not_Available', inplace=True)
        #any inf, nan value filled with zero
        df_test[num_cols].replace([np.inf, -np.inf,np.NaN,np.nan],0, inplace=True)            
        df_test.to_csv('./data/pre_processed_df_test_streamlit.csv', sep='\t', encoding='utf-8')
        return df_test
    def show_predicted_samples(data_predicted):
        will_default=[]
        for val in df_predicted['TARGET']:
            if val>=0.5:
                will_default.append('Yes')
            else:
                will_default.append('No')
        df_predicted['will_default?']=will_default
        st.subheader('Predicted samples....')
        df_predicted.rename(columns={'SK_ID_CURR': 'Application ID', 'TARGET': 'Probability'}, inplace=True)        
        st.write(df_predicted)
        return
    
menu = ["Select option","Raw Data","Preprocessed Data","Transformed Data"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Raw Data":
    st.subheader("Upload Raw Data...")
    data_file = st.file_uploader("CSV file",type=["csv"])
    if data_file is not None:
        file_details = {"filename":data_file.name, "filetype":data_file.type,"filesize":data_file.size}   
        st.write(file_details) 
        df_test_raw=pd.read_csv(data_file)
        if df_test_raw.shape[0]>9:
            df_test_raw=df_test_raw.head(10)

        if st.checkbox('Show raw data'):
            st.subheader("Raw Data...")
            st.write(df_test_raw)

        #preprocess data
        df_test_preprocessed = home_credit_streamlit.process_application_test(datafile=df_test_raw)
        df_test_vectorized=Home_credit_default_vectorization.vectorize_test_data(df_test_preprocessed) 
        df_predicted=Home_credit_LGBM.test_model(df_test_vectorized)
        home_credit_streamlit.show_predicted_samples(df_predicted)

        
elif choice == "Preprocessed Data":
    st.subheader("Upload preprocessed Data...")
    data_file = st.file_uploader("Upload Preprocessed CSV file",type=["csv"])
    if data_file is not None:
        file_details = {"filename":data_file.name, "filetype":data_file.type,"filesize":data_file.size}   
        st.write(file_details)     
        df_test_preprocessed = pd.read_csv(data_file, sep='\t', encoding='utf-8')
        df_test_preprocessed.drop(['Unnamed: 0'],axis=1,inplace=True)
        if st.checkbox('Show preprocessed data'):
            st.subheader("preprocessed Data...")
            st.dataframe(df_test_preprocessed)
        if df_test_preprocessed.shape[0]>9:
            df_test_preprocessed=df_test_preprocessed.head(10)
        df_test_vectorized=Home_credit_default_vectorization.vectorize_test_data(df_test_preprocessed)
        df_predicted=Home_credit_LGBM.test_model(df_test_vectorized)
        home_credit_streamlit.show_predicted_samples(df_predicted)

elif choice == "Transformed Data":
    st.subheader("Upload Transformed Data...")
    data_file = st.file_uploader("Upload Transformed pickle file",type=["pkl"])
    if data_file is not None:    
        with open('./data/df_test_vectorized_streamlit.pkl', 'rb') as f:
            df_test_vectorized = pickle.load(f)
        file_details = {"filename":data_file.name, "filetype":data_file.type,"filesize":data_file.size}   
        st.write(file_details)                 
        st.subheader('predicting data...')
        df_predicted=Home_credit_LGBM.test_model(df_test_vectorized)
        home_credit_streamlit.show_predicted_samples(df_predicted)


