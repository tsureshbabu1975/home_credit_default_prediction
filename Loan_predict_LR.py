#import packages to do EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import os

from torch import column_stack
plt.style.use('fivethirtyeight')
import gc
import matplotlib.font_manager

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.sparse import coo_matrix,hstack,vstack
import pickle
import joblib

from sklearn.metrics import confusion_matrix,roc_curve, auc
from sklearn.metrics import log_loss

from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import SGDClassifier
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)

class utility:
    def batch_predict(self,clf, data):
        # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
        # not the predicted outputs
        y_data_pred = []
        tr_loop = data.shape[0] - data.shape[0]%1000
        # consider you X_tr shape is 49041, then your tr_loop will be 49041 - 49041%1000 = 49000
        # in this for loop we will iterate unti the last 1000 multiplier
        for i in range(0, tr_loop, 1000):
            y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1])
        # we will be predicting for the last data points
        if data.shape[0]%1000 !=0:
            y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
        return y_data_pred
    #https://stackoverflow.com/questions/61748441/how-to-fix-the-values-displayed-in-a-confusion-matrix-in-exponential-form-to-nor
    def plot_confusionmatrix(self,y_tr,y_trpred,y_te,y_tepred):
        
        tn, fp, fn, tp = confusion_matrix(y_tr, np.round(y_trpred)).ravel()    
        #print('Training data tn-> {}, fp-> {}, fn-> {}, tp-> {}'.format(tn, fp, fn, tp), end=" ")
        #confusion matrix on training data 
        plt.figure(figsize=(10, 10))
        ax_tr = plt.subplot(221)
        cm_tr = confusion_matrix(y_tr, np.round(y_trpred))
        plt.title("Training data - Confusion Matrix")
        sns.heatmap(cm_tr, ax=ax_tr, fmt='d',cmap='YlGnBu',annot=True)
        # labels, title and ticks
        ax_tr.set_xlabel('Predicted labels');
        ax_tr.set_ylabel('True labels'); 
        ax_tr.set_ylim(2.0, 0)
        ax_tr.xaxis.set_ticklabels(['No','Yes']); 
        ax_tr.yaxis.set_ticklabels(['No','Yes']);
        
        #Confusion matrix on test data
        tn, fp, fn, tp = confusion_matrix(y_te, np.round(y_tepred)).ravel()    
        #print('Testing data tn-> {}, fp-> {}, fn-> {}, tp-> {}'.format(tn, fp, fn, tp), end=" ")
        
        ax_te = plt.subplot(222)
        cm_te = confusion_matrix(y_te, np.round(y_tepred))
        plt.title("Test data - Confusion Matrix")
        sns.heatmap(cm_te, ax=ax_te, fmt='d',cmap='YlGnBu',annot=True)
        # labels, title and ticks
        ax_te.set_xlabel('Predicted labels');
        ax_te.set_ylabel('True labels'); 
        ax_te.set_ylim(2.0, 0)
        ax_te.xaxis.set_ticklabels(['No','Yes']); 
        ax_te.yaxis.set_ticklabels(['No','Yes']);
        plt.show()
        return
    def draw_roccurve(self,y_tr,y_tr_pred,y_te,y_te_pred):
        #fpr,tpr,thresholds 
        fpr, tpr, thresholds = roc_curve(y_tr, np.array(y_tr_pred))
        #auc score train score
        auc_train = round(auc(fpr, tpr),5)
        plt.plot(fpr, tpr, label=" AUC train ="+str(auc_train))
        plt.plot([0, 1], [0, 1],'r--')
        
        fpr, tpr, thresholds = roc_curve(y_te, np.array(y_te_pred))
        #auc score test score
        auc_test = round(auc(fpr, tpr),5)
        plt.plot(fpr, tpr, label=" AUC test ="+str(auc_test))
        plt.plot([0, 1], [0, 1],'b--')
        
        plt.legend()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC" )
        plt.grid()
        plt.show()
        return    
    def model_fit(self,X_train,y_train):
        #dataset is highly imbalanced, hence calculate class weight
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
        d_class_weights = dict(enumerate(class_weights))
        
        clf=SGDClassifier(class_weight=d_class_weights,learning_rate='optimal', eta0=0.0,
                      loss='log', random_state=42,n_jobs=-1,verbose=0,
                     alpha=0.01,max_iter=400)
        
        #train the model with train dataset
        clf.fit(X=X_train, y=y_train)
        return clf
    
class model:
    def load_train_vectorized_df():
        with open('./data/df_train_vectorized.pkl', 'rb') as f:
            df = pickle.load(f)
        y =np.array(df['TARGET'])
        X =df.drop(['TARGET'], axis=1)
        features=X.columns.tolist()
        return np.array(X),y,features

    def train_model():
        obj_utility = utility()
        X,y,feature_names = model.load_train_vectorized_df()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y,random_state=42)
        #delete X and y to optimize the memory
        gc.enable()
        del X,y
        gc.collect()
        #pass the train and test to model and measure accuracy
        lr_model = obj_utility.model_fit(X_train,y_train)
        #predict
        y_train_pred = obj_utility.batch_predict(lr_model, X_train)    
        y_test_pred = obj_utility.batch_predict(lr_model, X_test)
     
        #confusion matrix
        obj_utility.plot_confusionmatrix(y_train,y_train_pred,y_test,y_test_pred)
        #draw roc curve
        obj_utility.draw_roccurve(y_train,y_train_pred,y_test,y_test_pred)
        
        gc.enable()
        del X_train,X_test
        gc.collect()
        
        #feature importance
        feature_importance = pd.DataFrame(lr_model.coef_[0], index=feature_names, 
                                          columns=['importance']).sort_values('importance', ascending=False)
        #store all features
        selected_features = list(feature_importance['importance'].index)
        '''
        #select top 200 features
        selected_features = list(feature_importance['importance'].head(1000).index)
        #reload
        X,y,feature_names = model.load_train_vectorized_df()
        #filter top 200 features and see auc curve
        df = pd.DataFrame(X,columns=feature_names)[selected_features]
        X_tr_selected, X_te_selected, y_train, y_test = train_test_split(np.array(df), y, test_size=0.33, stratify=y,random_state=42)
        gc.enable()
        del df
        gc.collect()
        #train the model again with selected 200 features
        lr_model_fine_tuned = obj_utility.model_fit(X_tr_selected,y_train)
        #save the model for future use
        #predict
        y_train_pred = obj_utility.batch_predict(lr_model_fine_tuned, X_tr_selected)    
        y_test_pred = obj_utility.batch_predict(lr_model_fine_tuned, X_te_selected)        
        #confusion matrix
        obj_utility.plot_confusionmatrix(y_train,y_train_pred,y_test,y_test_pred)
        #draw roc curve
        obj_utility.draw_roccurve(y_train,y_train_pred,y_test,y_test_pred)
        '''
        # save model
        joblib.dump(lr_model, './results/model_lr.pkl')
        joblib.dump(selected_features, './results/selected_features_lr.pkl')
        return

    def load_test_vectorized_df():
        with open('./data/df_test_vectorized.pkl', 'rb') as f:
            df = pickle.load(f)
        #SK_ID_CURR is required for submission            
        id =df['SK_ID_CURR']
        df.drop(columns=['SK_ID_CURR'],axis=1,inplace=True)
        return np.array(df),id
    
    def test_model():
        obj_utility = utility()
        X,id = model.load_test_vectorized_df()
        #load LGBM trained model
        lr_model = joblib.load(open('./results/model_lr.pkl', 'rb'))
        #predict
        y_test_pred = obj_utility.batch_predict(lr_model, X)
        submission = pd.DataFrame({'SK_ID_CURR': id.values, 'TARGET': y_test_pred})
        submission.to_csv('./results/submission_lr.csv')
        return

model.train_model()        
model.test_model()        

