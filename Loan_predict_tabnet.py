# imports necessary modules
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
import pandas as pd
import numpy as np
import tensorflow as tf

import os, gc, random, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_curve,roc_auc_score, confusion_matrix,auc
from pathlib import Path
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import random as rn
import pickle
import joblib
os.environ['PYTHONHASHSEED'] = '0'
## Set the random seed values to regenerate the model.
np.random.seed(0)
rn.seed(0)
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')
#clear the session 
tf.keras.backend.clear_session()
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
        
        ax_te = plt.subplot(222)
        cm_te = confusion_matrix(y_te, np.round(y_tepred))
        plt.title("Test data - Confusion Matrix")
        sns.heatmap(cm_te, ax=ax_te, fmt='d',cmap='YlGnBu',annot=True)
        # labels, title and ticks
        ax_te.set_ylabel('Predicted labels');
        ax_te.set_xlabel('True labels'); 
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
    
    def load_train_vectorized_df(self):
        with open('./data/df_train_vectorized_dl.pkl', 'rb') as f:
            df = pickle.load(f)
        y =np.array(df['TARGET'])
        X =df.drop(['TARGET'], axis=1)
        features=X.columns.tolist()
        return np.array(X),y,features
    
    def load_test_vectorized_df(self):
        with open('./data/df_test_vectorized_dl.pkl', 'rb') as f:
            df = pickle.load(f)
        #SK_ID_CURR is required for submission            
        id =df['SK_ID_CURR']
        df.drop(columns=['SK_ID_CURR'],axis=1,inplace=True)
        return np.array(df),id
    
    def load_utility_file(self):
        with open('./data/utilities_file.pkl', 'rb') as f:
            return pickle.load(f)
    def pretrain_model(self,X_train,tabnet_params):
        
        #max_epochs 30
        max_epochs = 30 if not os.getenv("CI", False) else 2
        #spilit X values into train valid and test
        X_tr,X_valid=train_test_split(X_train,test_size=0.33, random_state=42)        
        unsupervised_model = TabNetPretrainer(**tabnet_params)
        unsupervised_model.fit(X_train=X_tr,eval_set=[X_valid]
                            ,pretraining_ratio=0.8 ,max_epochs=max_epochs , patience=3
                            ,batch_size=4000, virtual_batch_size=256,num_workers=0
                            ,drop_last=False)
        #save unsupervised model
        unsupervised_model.save_model('./results/test_pretrain')        
        return unsupervised_model
    
    def model_fit(self,X_train,y_train,unsupervised_model,tabnet_params):
        
        #max_epochs 30
        max_epochs = 30 if not os.getenv("CI", False) else 2        
        X_tr,X_valid,y_tr,y_valid =train_test_split(X_train,y_train,test_size=0.33,stratify=y_train, random_state=42)
        clf = TabNetClassifier(**tabnet_params)
        model_history=clf.fit(X_train=X_tr, y_train=y_tr, 
                                eval_set=[(X_tr, y_tr), (X_valid,y_valid)],
                                eval_name=['train', 'valid'], eval_metric=['auc'], 
                                from_unsupervised=unsupervised_model,
                                max_epochs=max_epochs , patience=3,
                                batch_size=4000, virtual_batch_size=256,
                                num_workers=0,
                                drop_last=False,
                                weights=1 # No sampling
                            )
        #save unsupervised model
        clf.save_model('./results/tabnet_trained_model')
        return clf,model_history
        
class TabNet:
    def train_model():
        obj_utility=utility()
        #load utility file contains categorical feature index position,
        #categorical dimension and categorical embedding dimensions
        d_utility=obj_utility.load_utility_file()        
        #load vectorized train data
        X,y,feature_names = obj_utility.load_train_vectorized_df()
        #TabNet hyper parameters
        #n_d [Width of the decision prediction layer.] = 8 -64 default is 8
        #n_a [Width of the attention embedding for each mask.] = 8 -64 default is 8
        #n_steps : [Number of steps in the architecture (usually between 3 and 10)]. default=3
        #gamma : [This is the coefficient for feature reusage in the masks.]. default=1.3.Values range from 1.0 to 2.0.
        #cat_idxs :(default=[] - Mandatory for embeddings).List of categorical features indices.
        #cat_dims :(default=[] - Mandatory for embeddings). List of categorical features number of modalities
        #cat_emb_dim : list of int. List of embeddings size for each categorical features. (default =1)
        #n_independent : (default=2).[Number of independent Gated Linear Units layers at each step. Usual values range from 1 to 5.]
        #n_shared : int (default=2). [Number of shared Gated Linear Units at each step Usual values range from 1 to 5].
        #momentum : [Momentum for batch normalization, typically ranges from 0.01 to 0.4 ] (default=0.02)
        #clip_value : 0.95 [If a float is given this will clip the gradient at clip_value.]
        #lambda_sparse : (default = 1e-3): [This is the extra sparsity loss coefficient as proposed in the original paper]
        #optimizer_fn : torch.optim (default=torch.optim.Adam)
        #optimizer_params: dict (default=dict(lr=2e-2))
        #scheduler_fn : torch.optim.lr_scheduler (default=None)
        tabnet_params = {
                            'n_d': 24, 'n_a': 24, 'n_steps': 5,'gamma':1.3, 
                            'cat_idxs':list(d_utility['cat_indexes']),
                            'cat_dims':list(d_utility['cat_dimensions']), 
                            'cat_emb_dim':list(d_utility['cat_embedding_dim']),
                            'n_independent': 2, 
                            'n_shared': 2, 'epsilon': 1e-15, 
                            'momentum': 0.15, 'verbose': 1, 
                            'optimizer_fn': torch.optim.Adam, 
                            'optimizer_params': dict(lr=2e-2), 
                            'scheduler_fn': torch.optim.lr_scheduler.StepLR, 
                            'scheduler_params': {"step_size":20,'gamma':0.95}, 
                            'mask_type': 'entmax', #sparsemax
                            'input_dim': X.shape[1], 
                            'output_dim': 2, 
                            'device_name': 'auto',
                            'lambda_sparse': 1e-4,
                            'clip_value' : 2.0
                        }
        
        #invoke pretrained model
        unsupervised_model=obj_utility.pretrain_model(X,tabnet_params)
        #spilit X values into train and test
        X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.33,stratify=y,random_state=42)
        #model_fit
        tabnet_clf,model_history=obj_utility.model_fit(X_train,y_train,unsupervised_model,tabnet_params)
        return
    
    def train_predict():
        obj_utility = utility()
        X,y,feature_names = obj_utility.load_train_vectorized_df()        
        #spilit X values into train and test
        X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.33,stratify=y,random_state=42)
        tabnet_clf = TabNetClassifier()
        tabnet_clf.load_model('./results/tabnet_trained_model.zip')    
        #predict
        y_train_pred = obj_utility.batch_predict(tabnet_clf, X_train)    
        y_test_pred = obj_utility.batch_predict(tabnet_clf, X_test)
        #confusion matrix
        obj_utility.plot_confusionmatrix(y_train,y_train_pred,y_test,y_test_pred)
        #draw roc curve
        obj_utility.draw_roccurve(y_train,y_train_pred,y_test,y_test_pred)
        #feature importance
        #feature_importance = pd.DataFrame(tabnet_clf.feature_importances_, index=feature_names, 
        #                                  columns=['importance']).sort_values('importance', ascending=False)
        #select top 200 features
        #selected_features = list(feature_importance['importance'].index)
        gc.enable()
        del X_train,X_test
        gc.collect()
        return        

    def test_predict():
        obj_utility = utility()
        X,id = obj_utility.load_test_vectorized_df()
        tabnet_clf = TabNetClassifier()
        tabnet_clf.load_model('./results/tabnet_trained_model.zip')  
        #predict
        y_test_pred = obj_utility.batch_predict(tabnet_clf, X)
        submission = pd.DataFrame({'SK_ID_CURR': id.values, 'TARGET': y_test_pred})
        submission.to_csv('./results/submission_tabnet.csv')
        return
TabNet.train_model()       
TabNet.train_predict() 
TabNet.test_predict()    