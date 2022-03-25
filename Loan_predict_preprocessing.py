#import packages to do preprocessing
import os
import gc

from unicodedata import decimal
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

class preprocess:
    #Method reduce_memory_usage will optimize the file size
    #Credit :- https://www.kaggle.com/rinnqd/reduce-memory-usage
    def optimize_memory_usage(self, df):
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2
        #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
        return df

    def drop_columns(self, df,cols):
        #Step 2: Columns to be removed based on the EDA section.
        #drop columns from the dataset
        df.drop(cols,axis=1,inplace=True)
        return df

    #function to convert negative values to postive and convert them into years. 
    def convert_negative_days_year(self, days_negative):
        #convert negative values to positive, and divide by 365 days to convert years
        #finally rounded to 1 decimal.
        return np.round(abs(days_negative)/365,1)

    #Try to add some more features domain based
    # Credit :- https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction#Feature-Engineering
    def extract_domainbased_features(self, df):
        epsilon=0.001 # to avoid div/0 error 
        df1 = pd.DataFrame()
        df1['SK_ID_CURR']=df['SK_ID_CURR']
        df1['pc_Credit_Income'] = np.round((df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL']+epsilon)),4)
        df1['pc_Annuity_Income'] = np.round((df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL']+epsilon)),4)
        df1['pc_Credit_Annuity'] = np.round((df['AMT_CREDIT'] / (df['AMT_ANNUITY']+epsilon)),4)
        df1['Credit_Goods_Diff'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
        df1['pc_Loan_Value']=np.round((df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE']+epsilon)),4)
        df1['CREDIT_TERM'] = np.round((df['AMT_ANNUITY'] / (df['AMT_CREDIT']+epsilon)),4)
        df1['pc_Employment_Age'] = np.round((df['Employment'] / (df['Age']+epsilon)),4)
        return df1

    #some of the features datatypes needs to be converted to float 64 and rounded to 4 decimals 
    def convert_feature_dtype_and_round(self, df, list_of_features):
        for feature in list_of_features:
            df[feature]=df[feature].round(decimals=4).astype(np.float64)
        return df
    #some of the features datatypes needs to be converted to float 64 
    def convert_feature_dtype(self, df, list_of_features):
        for feature in list_of_features:
            df[feature]=df[feature].astype(np.float64)
        return df

    #extract polinomial features for given original features
    def extract_polinomial_features(self, df, list_of_features):
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        poly_features = df[list_of_features]
        # Need to impute missing values
        poly_features = imputer.fit_transform(poly_features)
        # Create the polynomial object with specified degree
        poly_transformer = PolynomialFeatures(degree = 2)
        # Train the polynomial features
        poly_features= poly_transformer.fit_transform(poly_features)
        poly_features = pd.DataFrame(poly_features, 
                            columns = poly_transformer.get_feature_names(
                                            input_features = list_of_features))
        #add ID column to poly features to merge the polynomial features
        poly_features['SK_ID_CURR'] = df['SK_ID_CURR']
        
        # Merge polynomial features into dataframe
        df = df.merge(poly_features, on = 'SK_ID_CURR', how = 'left')
        return df

    # numeric aggregation on other dataframes and group by ID
    # credit:https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
    def FE_numerical_features(self,df, group_var, df_name):
        """Aggregates the numeric values in a dataframe. This can
        be used to create features for each instance of the grouping variable.
        
        Parameters
        --------
            df (dataframe): 
                the dataframe to calculate the statistics on
            group_var (string): 
                the variable by which to group df
            df_name (string): 
                the variable used to rename the columns
            
        Return
        --------
            agg (dataframe): 
                a dataframe with the statistics aggregated for 
                all numeric columns. Each instance of the grouping variable will have 
                the statistics (mean, min, max, sum; currently supported) calculated. 
                The columns are also renamed to keep track of features created.
        
        """
        # Remove id variables other than grouping variable
        for col in df:
            if col != group_var and 'SK_ID' in col:
                df = df.drop(columns = col)
                
        group_ids = df[group_var]
        numeric_df = df.select_dtypes('number')
        numeric_df[group_var] = group_ids

        # Group by the specified variable and calculate the statistics
        agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

        # Need to create new column names
        columns = [group_var]

        # Iterate through the variables names
        for var in agg.columns.levels[0]:
            # Skip the grouping variable
            if var != group_var:
                # Iterate through the stat names
                for stat in agg.columns.levels[1][:-1]:
                    # Make a new column name for the variable and stat
                    columns.append('%s_%s_%s' % (df_name, var, stat))

        agg.columns = columns
        return agg

    # calculate the counts and normalized counts of each category for all categorical variables in the dataframe.
    # very similar to numeric_aggregate
    ## credit:https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering

    def FE_categorical_features(self,df, group_var, df_name):
        """Computes counts and normalized counts for each observation
        of `group_var` of each unique category in every categorical variable
        
        Parameters
        --------
        df : dataframe 
            The dataframe to calculate the value counts for.
            
        group_var : string
            The variable by which to group the dataframe. For each unique
            value of this variable, the final dataframe will have one row
            
        df_name : string
            Variable added to the front of column names to keep track of columns

        
        Return
        --------
        categorical : dataframe
            A dataframe with counts and normalized counts of each unique category in every categorical variable
            with one row for every unique value of the `group_var`.
            
        """
        
        # Select the categorical columns
        categorical = pd.get_dummies(df.select_dtypes('object'))

        # Make sure to put the identifying id on the column
        categorical[group_var] = df[group_var]

        # Groupby the group var and calculate the sum and mean
        categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
        
        column_names = []
        
        # Iterate through the columns in level 0
        for var in categorical.columns.levels[0]:
            # Iterate through the stats in level 1
            for stat in ['count', 'count_norm']:
                # Make a new column name
                column_names.append('%s_%s_%s' % (df_name, var, stat))
        
        categorical.columns = column_names
        
        return categorical

    #except train and test dataset, other datasets are common.
    #before feature transformation, do the imputation on missing values
    #for categorical values, based on the dataset (Train / Test), imputes most categorical values
    #in case of all data is missing, will be nan value will be filled with "Data_Not_Available"
    def fill_mostfrequent_value(self,df):
        cat_cols=df.select_dtypes(include=object).columns #categorical values
        for col in cat_cols:
            #fill most frequent value
            df[col].fillna(df[col].value_counts(dropna=True).index[0], inplace=True)
        return pd.DataFrame(df)
    #for numerical values, filled with median value of that particular feature. 
    #if all values are null, then filled with zero.
    def fill_median_value(self,df):
        num_cols=df.select_dtypes(exclude=object).columns
        for col in num_cols:
            df[col].fillna(df[col].median(axis=0,skipna = True),inplace=True)
        return pd.DataFrame(df)
class bureau:
    #Bureau dataset is 1-M hence after FE, group by application id
    def merge_bureau_df(self,df):
        obj = preprocess()
        #Load bureau dataset and optimize the size
        df_bureau = obj.optimize_memory_usage(pd.read_csv('./data/bureau.csv', encoding= 'unicode_escape'))
        #filter rows only that matches with df
        df_bureau.merge(df['SK_ID_CURR'], on='SK_ID_CURR', how='inner')
        #find categorical columns in bureau dataset
        cat_columns=df_bureau.select_dtypes(include=object).columns.to_list()
        cat_columns.append('SK_ID_CURR')
        #rename columns as it is similar to train/test dataset
        df_bureau.rename(columns={"AMT_ANNUITY": "BUREAU_AMT_ANNUITY"}, inplace = True)
        #convert to positive values 
        df_bureau['DAYS_CREDIT']=-df_bureau['DAYS_CREDIT']
        df_bureau['DAYS_ENDDATE_FACT']=-df_bureau['DAYS_ENDDATE_FACT']
        #fill missing value using simple imputation
        df_bureau= obj.fill_mostfrequent_value(df_bureau)
        df_bureau =obj.fill_median_value(df_bureau)
        #feature transformation on numerical columns
        #merge df_bureau categorical columns with train since it is multiple take the 1 record per I
        df_bureau_categorical_group =df_bureau[cat_columns].groupby('SK_ID_CURR').head(1).reset_index(drop=True)
        # Group by the applicant id, calculate aggregation statistics
        df_bureau_agg = obj.FE_numerical_features(df_bureau.drop(columns = ['SK_ID_BUREAU']),group_var = 'SK_ID_CURR', df_name = 'bureau')
        #feature transformation on categorical columns
        df_bureau_categorical = obj.FE_categorical_features(df=df_bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
        #finally merge
        df =df.merge(df_bureau_categorical_group,on='SK_ID_CURR', how='left' )
        df =df.merge(df_bureau_agg, on = 'SK_ID_CURR', how = 'left')
        df =df.merge(df_bureau_categorical, on = 'SK_ID_CURR', how = 'left')
        #After merging, there may be null values. Hence fill with  most_frequent values on categorical
        #and for numerical, fill with median 
        df_bureau = obj.fill_mostfrequent_value(df_bureau)
        df_bureau = obj.fill_median_value(df_bureau)      
        #After merging, there may be null values. Hence fill with  most_frequent values on categorical
        #and for numerical, fill with median 
        df = obj.fill_mostfrequent_value(df)
        df = obj.fill_median_value(df)
          
        #return merged dataset and bureau dataset
        return df,df_bureau
    #bureau balance dataset is one level further down by SK_ID_BUREAU
    #hence two level grouping is required to merge with train/test dataset
class bureau_balance:
    def merge_bureau_balance_df(self,df,df_bureau):
        obj = preprocess()
         #Load bureau dataset and optimize the size
        df_bureau_balance =  obj.optimize_memory_usage(pd.read_csv('./data/bureau_balance.csv', encoding= 'unicode_escape'))
        #filter rows only that matches with df
        df_bureau_balance.merge(df_bureau['SK_ID_BUREAU'], on='SK_ID_BUREAU', how='inner')
        #find categorical columns in bureau dataset
        cat_columns=df_bureau_balance.select_dtypes(include=object).columns.to_list()
        #since overdue doesn't have negative values, convert them into positive
        df_bureau_balance['MONTHS_BALANCE'] = -df_bureau_balance['MONTHS_BALANCE']
        # fill missing values
        df_bureau_balance=obj.fill_mostfrequent_value(df_bureau_balance)
        df_bureau_balance=obj.fill_median_value(df_bureau_balance)
        #feature transformation on numerical columns
        df_bureau_balance_agg = obj.FE_numerical_features(df=df_bureau_balance,group_var = 'SK_ID_BUREAU', df_name = 'bureau_bal')
        #Add Categorical field numerical counts
        df_bureau_balance_categorical = obj.FE_categorical_features(df=df_bureau_balance, 
                                                                                group_var = 'SK_ID_BUREAU', df_name = 'bureau_bal')
        # Dataframe grouped by the loan
        bureau_by_loan = df_bureau_balance_agg.merge(df_bureau_balance_categorical, right_index = True, 
                                             left_on = 'SK_ID_BUREAU', how = 'left')
        # Merge to include the SK_ID_CURR
        bureau_by_loan = df_bureau[['SK_ID_BUREAU', 'SK_ID_CURR']]\
                            .merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left').reset_index(drop=True)
        #drop SK_ID_BUREAU column and aggreate at client level
        bureau_by_loan.drop(columns = ['SK_ID_BUREAU'],axis=1,inplace=True)
        # Aggregate the stats for each client
        bureau_balance_by_client = obj.FE_numerical_features(df=bureau_by_loan,group_var = 'SK_ID_CURR', df_name = 'client')
        # Merge with the df
        df = df.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left').reset_index(drop=True)
        #After merging, there may be null values. Hence fill with  most_frequent values on categorical
        #and for numerical, fill with median 
        df = obj.fill_mostfrequent_value(df)
        df = obj.fill_median_value(df)
        #return df merged with bureau balance dataset
        return df       
class previous_loan:
    def merge_prev_loan_count(self,df,df_bureau):
        #find number of previous loans per applicant, this will also help us to understand how many loans applied by client in the past
        prev_loan_cnt = df_bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().\
                                                rename(columns = {'SK_ID_BUREAU': 'prev_loan_cnt'})
        #merge previous loan counts to train dataset
        df = df.merge(prev_loan_cnt, on = 'SK_ID_CURR', how = 'left')
        return df    
class previous_application:
    def merge_prev_app_df(self,df):
        obj = preprocess()
        #Load bureau dataset and optimize the size
        df_prev_app =obj.optimize_memory_usage(pd.read_csv('./data/previous_application.csv', encoding= 'unicode_escape'))
        #filter rows only that matches with df
        df_prev_app.merge(df['SK_ID_CURR'], on='SK_ID_CURR', how='inner')
        #rename columns
        df_prev_app.rename(columns={"NAME_CONTRACT_TYPE": "pre_app_NAME_CONTRACT_TYPE"}, inplace = True)
        df_prev_app.rename(columns={"AMT_ANNUITY": "pre_app_AMT_ANNUITY"}, inplace = True)
        df_prev_app.rename(columns={"AMT_CREDIT": "pre_app_AMT_CREDIT"}, inplace = True)
        df_prev_app.rename(columns={"AMT_GOODS_PRICE": "pre_app_AMT_GOODS_PRICEE"}, inplace = True)
        df_prev_app.rename(columns={"WEEKDAY_APPR_PROCESS_START": "pre_app_WEEKDAY_APPR_PROCESS_START"}, inplace = True)
        df_prev_app.rename(columns={"HOUR_APPR_PROCESS_START": "pre_app_HOUR_APPR_PROCESS_START"}, inplace = True)
        df_prev_app.rename(columns={"NAME_TYPE_SUITE": "pre_app_NAME_TYPE_SUITE"}, inplace = True)
        #convert negative values to zero as there is not concept of negative down payment
        df_prev_app.loc[df_prev_app['AMT_DOWN_PAYMENT'] <0,'AMT_DOWN_PAYMENT']=0
        #DAYS DECISION will be +ve days
        df_prev_app['DAYS_DECISION']=-df_prev_app['DAYS_DECISION']
        cat_columns=df_prev_app.select_dtypes(include=object).columns.to_list()
        cat_columns.append('SK_ID_CURR')
        # fill missing values
        df_prev_app=obj.fill_mostfrequent_value(df_prev_app)
        df_prev_app=obj.fill_median_value(df_prev_app)
        #merge previous application categorical columns with train since it is multiple take the 1 record per ID
        df_Pre_App_cat_group =df_prev_app[cat_columns].groupby('SK_ID_CURR').head(1).reset_index(drop=True)
        # Group by the applicant id, calculate aggregation statistics for numerical features
        df_prev_app_numerical_agg = obj.FE_numerical_features(df_prev_app.drop(columns = ['SK_ID_PREV']),
                                                        group_var = 'SK_ID_CURR', df_name = 'pre_app').reset_index(drop=True)
        #FE on categorical columns
        df_prev_app_cat_agg = obj.FE_categorical_features(df=df_prev_app,group_var = 'SK_ID_CURR', df_name = 'pre_app').reset_index()
        #merge with df
        df=df.merge(df_Pre_App_cat_group,on='SK_ID_CURR', how='left' )
        df=df.merge(df_prev_app_numerical_agg,on='SK_ID_CURR', how='left')
        df=df.merge(df_prev_app_cat_agg,on='SK_ID_CURR', how='left' )
        #After merging, there may be null values. Hence fill with  most_frequent values on categorical
        #and for numerical, fill with median 
        df = obj.fill_mostfrequent_value(df)
        df = obj.fill_median_value(df)        
        return df
class monthly_cash:
    def merge_monthly_cash_df(self,df):
        obj = preprocess()
        #load and optimize cash balance dataset
        df_cash_balance = obj.optimize_memory_usage(pd.read_csv('./data/POS_CASH_balance.csv', encoding= 'unicode_escape'))
        #rename the columns
        df_cash_balance.rename(columns={"NAME_CONTRACT_STATUS": "Cash_Bal_NAME_CONTRACT_STATUS"}, inplace = True)
        df_cash_balance.rename(columns={"MONTHS_BALANCE": "Cash_Bal_MONTHS_BALANCE"}, inplace = True)
        #categorical columns
        cat_columns=df_cash_balance.select_dtypes(include=object).columns.to_list()
        cat_columns.append('SK_ID_CURR')
        #convert balance into positive values
        df_cash_balance['Cash_Bal_MONTHS_BALANCE']=-df_cash_balance['Cash_Bal_MONTHS_BALANCE']
        # fill missing values
        df_cash_balance=obj.fill_mostfrequent_value(df_cash_balance)
        df_cash_balance=obj.fill_median_value(df_cash_balance)
        #merge previous application categorical columns with train since it is multiple take the 1 record per ID
        df_Cash_Bal_cat_group =df_cash_balance[cat_columns].groupby('SK_ID_CURR').head(1).reset_index(drop=True)
        # Group by the applicant id, calculate aggregation statistics
        df_cash_bal_numerical_agg = obj.FE_numerical_features(df_cash_balance.drop(columns = ['SK_ID_PREV']),
                                                    group_var = 'SK_ID_CURR', df_name = 'cash_bal').reset_index(drop=True)
        #categorical features
        df_cash_bal_cat_agg = obj.FE_categorical_features(df=df_cash_balance, 
                                                                group_var = 'SK_ID_CURR', df_name = 'cash_bal').reset_index()
        #merge with df
        df=df.merge(df_Cash_Bal_cat_group,on='SK_ID_CURR', how='left')
        df=df.merge(df_cash_bal_numerical_agg,on='SK_ID_CURR', how='left' )
        df=df.merge(df_cash_bal_cat_agg,on='SK_ID_CURR', how='left' )
        #After merging, there may be null values. Hence fill with  most_frequent values on categorical
        #and for numerical, fill with median 
        df = obj.fill_mostfrequent_value(df)
        df = obj.fill_median_value(df)        
        return df
class monthly_credit:
    def merge_monthly_credit_df(self,df):
        obj = preprocess()
        df_CCard_balance = obj.optimize_memory_usage(pd.read_csv('./data/credit_card_balance.csv', encoding= 'unicode_escape'))
        #convert negative values to positive
        df_CCard_balance['MONTHS_BALANCE']=-df_CCard_balance['MONTHS_BALANCE']
        #convert negative values to zero as there is not concept of negative balance amount
        df_CCard_balance.loc[df_CCard_balance['AMT_BALANCE'] <0,'AMT_BALANCE']=0
        df_CCard_balance.loc[df_CCard_balance['AMT_DRAWINGS_ATM_CURRENT'] <0,'AMT_DRAWINGS_ATM_CURRENT']=0
        df_CCard_balance.loc[df_CCard_balance['AMT_DRAWINGS_CURRENT'] <0,'AMT_DRAWINGS_CURRENT']=0
        df_CCard_balance.loc[df_CCard_balance['AMT_RECEIVABLE_PRINCIPAL'] <0,'AMT_RECEIVABLE_PRINCIPAL']=0
        df_CCard_balance.loc[df_CCard_balance['AMT_RECIVABLE'] <0,'AMT_RECIVABLE']=0
        df_CCard_balance.loc[df_CCard_balance['AMT_TOTAL_RECEIVABLE'] <0,'AMT_TOTAL_RECEIVABLE ']=0 
        #since NAME_CONTRACT_STATUS is repeated. rename this column
        df_CCard_balance.rename(columns = {'NAME_CONTRACT_STATUS':'CCard_NAME_CONTRACT_STATUS'}, inplace = True)
        df_CCard_balance.rename(columns = {'MONTHS_BALANCE':'CCard_MONTHS_BALANCE'}, inplace = True)
        #categorical columns
        cat_columns=df_CCard_balance.select_dtypes(include=object).columns.to_list()
        cat_columns.append('SK_ID_CURR')
        # fill missing values
        df_CCard_balance=obj.fill_mostfrequent_value(df_CCard_balance)
        df_CCard_balance=obj.fill_median_value(df_CCard_balance)
        #merge previous application categorical columns with train since it is multiple take the 1 record per ID
        df_CCard_Bal_cat_group =df_CCard_balance[cat_columns].groupby('SK_ID_CURR').head(1).reset_index(drop=True)
        # Group by the applicant id, calculate aggregation statistics
        df_CCard_bal_numerical_agg=obj.FE_numerical_features(df_CCard_balance.drop(columns = ['SK_ID_PREV']),
                                                    group_var = 'SK_ID_CURR', df_name = 'CC_bal').reset_index(drop=True)
        #group by categorical columns
        df_CCard_bal_cat_agg=obj.FE_categorical_features(df=df_CCard_balance, 
                                                                group_var = 'SK_ID_CURR', df_name = 'CC_bal').reset_index()
        #merge with df
        df=df.merge(df_CCard_Bal_cat_group,on='SK_ID_CURR', how='left')
        df=df.merge(df_CCard_bal_numerical_agg,on='SK_ID_CURR', how='left' )
        df=df.merge(df_CCard_bal_cat_agg,on='SK_ID_CURR', how='left' )
        #After merging, there may be null values. Hence fill with  most_frequent values on categorical
        #and for numerical, fill with median 
        df = obj.fill_mostfrequent_value(df)
        df = obj.fill_median_value(df)
        return df       
class installment_payment:
    def merge_install_payments_df(self,df):
        obj = preprocess()
        #load and optimize cash balance dataset
        df_Installment_payments=obj.optimize_memory_usage(pd.read_csv('./data/installments_payments.csv', encoding= 'unicode_escape'))
        #convert negative values to positive
        df_Installment_payments['DAYS_INSTALMENT']=-df_Installment_payments['DAYS_INSTALMENT']
        df_Installment_payments['DAYS_ENTRY_PAYMENT']=-df_Installment_payments['DAYS_ENTRY_PAYMENT']
        # fill missing values
        df_Installment_payments =obj.fill_median_value(df_Installment_payments)
        # Group by the applicant id, calculate aggregation statistics
        df_Install_payment_numerical_agg =obj.FE_numerical_features(df_Installment_payments.drop(columns = ['SK_ID_PREV']),
                                              group_var = 'SK_ID_CURR', df_name = 'Install_pay').reset_index(drop=True)
        df=df.merge(df_Install_payment_numerical_agg,on='SK_ID_CURR', how='left' )
        #After merging, there may be null values. Hence fill with  most_frequent values on categorical
        #and for numerical, fill with median 
        df = obj.fill_mostfrequent_value(df)
        df = obj.fill_median_value(df)        
        return df

class home_credit_default:
    def process_data(df):
        obj_process = preprocess()
        #optimize the raw dataset
        df = obj_process.optimize_memory_usage(df)
        #below columns are removed as it is highly correlated with other columns. Refer EDA 
        cols=['CNT_FAM_MEMBERS','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
                'LIVE_REGION_NOT_WORK_REGION', 'LIVE_CITY_NOT_WORK_CITY','ELEVATORS_AVG', 'ENTRANCES_AVG','LIVINGAREA_AVG', 
                'APARTMENTS_MODE','YEARS_BEGINEXPLUATATION_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE','FLOORSMAX_MODE',
                'LIVINGAREA_MODE', 'APARTMENTS_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI', 'LIVINGAREA_MEDI', 
                'HOUSETYPE_MODE', 'TOTALAREA_MODE','YEARS_BEGINEXPLUATATION_MEDI','FLOORSMAX_AVG',
                'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
                'BASEMENTAREA_MODE','BASEMENTAREA_MEDI','YEARS_BUILD_MODE','YEARS_BUILD_MEDI','FLOORSMIN_MODE','FLOORSMIN_MEDI',
                'LIVINGAPARTMENTS_MODE','LIVINGAPARTMENTS_MEDI','NONLIVINGAPARTMENTS_MODE','NONLIVINGAPARTMENTS_MEDI',
                'COMMONAREA_MEDI','COMMONAREA_MODE','LANDAREA_MODE','LANDAREA_MEDI','NONLIVINGAREA_MODE','NONLIVINGAREA_MEDI',
                'FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_7','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
                'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21'
                ]
        #drop highly correlated fields from dataset
        df=obj_process.drop_columns(df,cols)
        #spilit categorical and numerical fields
        categorical_columns=df.select_dtypes(include=object).columns.to_list()
        numerical_columns=df.select_dtypes(exclude=object).columns.to_list()
        #remove outlier from DAYS_EMPLOYED field
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        #convert  the following fields into positive and divide by 365 to convert year
        df['Age'] = obj_process.convert_negative_days_year(df['DAYS_BIRTH'])
        df['Employment'] =  obj_process.convert_negative_days_year(df['DAYS_EMPLOYED'])
        df['Phone_change_Years'] =  obj_process.convert_negative_days_year(df['DAYS_LAST_PHONE_CHANGE'])
        df['Registration_Years'] =  obj_process.convert_negative_days_year(df['DAYS_REGISTRATION'])
        df['Id_Publish_Years'] =  obj_process.convert_negative_days_year(df['DAYS_ID_PUBLISH'])
        #The below columns are transformed. hence remove it from the dataset
        cols=['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_LAST_PHONE_CHANGE','DAYS_REGISTRATION','DAYS_ID_PUBLISH']
        df=obj_process.drop_columns(df,cols)
        
        df = obj_process.fill_mostfrequent_value(df)
        df = obj_process.fill_median_value(df)
        #domain based FE
        df = df.merge(obj_process.extract_domainbased_features(df), on = 'SK_ID_CURR', how = 'left')
        #convert following features datatype and round it to 4 decimals uniformly
        cols = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
        df = obj_process.convert_feature_dtype_and_round(df=df,list_of_features=cols)
        #convert following features dataype only
        cols = ['Age','Employment']
        df = obj_process.convert_feature_dtype(df=df,list_of_features=cols)
        #extract polinomial features for the following colums and combine to given train/test dataset
        cols=['EXT_SOURCE_1','EXT_SOURCE_2', 'EXT_SOURCE_3', 'Age', 'Employment']
        df = obj_process.extract_polinomial_features(df,cols)
        #merge bureau dataset to given train/test dataset
        obj_bureau = bureau()
        df,df_bureau = obj_bureau.merge_bureau_df(df)
        #merge bureu balance dataset to 
        obj_bureau_balance = bureau_balance()
        df = obj_bureau_balance.merge_bureau_balance_df(df=df,df_bureau=df_bureau)
        #merge previous loan count
        obj_prev_loan_cnt = previous_loan()
        df = obj_prev_loan_cnt.merge_prev_loan_count(df=df,df_bureau=df_bureau)
        #delete df_bureau dataset which is no longer is required.
        gc.enable()
        del df_bureau
        gc.collect()
        #merge previous application dataset
        obj_prev_app = previous_application()
        df = obj_prev_app.merge_prev_app_df(df)
        #merge monthly cash dataset
        obj_cash = monthly_cash()
        df = obj_cash.merge_monthly_cash_df(df)
        #merge monthly credit dataset
        obj_credit=monthly_credit()
        df = obj_credit.merge_monthly_credit_df(df)
        #merge installment payments
        obj_ins_pay = installment_payment()
        df = obj_ins_pay.merge_install_payments_df(df)
        #finally optimize the final dataset
        df = obj_process.optimize_memory_usage(df)
        return df
    #preprocess train data
    def process_application_train():
        #optimize file size of application train data
        df_train =  pd.read_csv('./data/application_train.csv', encoding= 'unicode_escape')
        df_train = home_credit_default.process_data(df=df_train)
        #any inf, nan value filled with zero
        #find categorical and numerical fields
        cat_cols=df_train.select_dtypes(include=object).columns.to_list()
        num_cols=df_train.select_dtypes(exclude=object).columns.to_list()
        
        df_train[cat_cols].replace([np.inf, -np.inf,np.NaN,np.nan], 'Data_Not_Available', inplace=True)
        df_train[num_cols].replace([np.inf, -np.inf,np.NaN,np.nan],0, inplace=True)        
        df_train.to_csv('./data/pre_processed_df_train.csv', sep='\t', encoding='utf-8')
        print(df_train.shape)
    #preprocess test data    
    def process_application_test():
        #optimize file size of application train data
        df_test =  pd.read_csv('./data/application_test.csv', encoding= 'unicode_escape')
        df_test = home_credit_default.process_data(df=df_test)
        #find categorical and numerical fields
        cat_cols=df_test.select_dtypes(include=object).columns.to_list()

        num_cols=df_test.select_dtypes(exclude=object).columns.to_list()
        #categorical values fill data_not_available        
        df_test[cat_cols].replace([np.inf, -np.inf,np.NaN,np.nan], 'Data_Not_Available', inplace=True)
        #any inf, nan value filled with zero
        df_test[num_cols].replace([np.inf, -np.inf,np.NaN,np.nan],0, inplace=True)            
        df_test.to_csv('./data/pre_processed_df_test_streamlit.csv', sep='\t', encoding='utf-8')
        return df_test    

#home_credit_default.process_application_train()
#home_credit_default.process_application_test()
