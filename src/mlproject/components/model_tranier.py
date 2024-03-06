import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from yellowbrick.features import FeatureImportances
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object,evaluate_models
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    
    def feature_selection(self,X_train,X_test):
        logging.info("Performing Correlation")
        
        threshold=0.7
        col_corr=set()
        corr_matrix=X_train.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
               if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
        X_train_features=X_train.drop(col_corr,axis=1)
        X_test_features=X_test.drop(col_corr,axis=1)
        print(X_train_features)
        return X_train_features,X_test_features
    

    def feature_importance(self,selected_model,Xtrain,Xtest,ytrain):
        viz= FeatureImportances(selected_model, topn=25)
        viz.fit(Xtrain, ytrain)
        X_train_returned=Xtrain[viz.features_]
        X_test_returned=Xtest[viz.features_]

        return X_train_returned,X_test_returned
    
    def images(data,classifier):
        fn=data.feature_names
        cn=data.target_names
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
        tree.plot_tree(classifier.values.estimators_[2],
        feature_names = fn, 
        class_names=cn,
        filled = True)
        fig.savefig('rf_individualtree.png')

        
    def initiate_model_trainer(self,train_data_path,test_data_path):
        try: 
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info("Split training and test input data")
            X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
            X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
            #calling feature_selection function 
            X_train_features,X_test_features=self.feature_selection(X_train,X_test)
            
            #Random Forest Model
            model=RandomForestClassifier()
            #HyperparameterTuning
            param_dist = {'n_estimators': randint(100,110),      
                          'max_depth':  list(range(10, 100, 10)),  
                          'min_samples_split': randint(2, 20),      
                          'min_samples_leaf': randint(1, 20),
                           'criterion':['gini', 'entropy','log_loss'] }
            
            logging.info("Training model")
            
            #calling evaluate_models from utils.py
            model_returned=evaluate_models(X_train=X_train_features,y_train=y_train,X_test=X_test_features,y_test=y_test,
                                             model=model,param_dist=param_dist)
            print(model_returned)

            X_train_selected,X_test_selected=self.feature_importance(model_returned,X_train_features,X_test_features,y_train)
            model_returned.fit(X_train_selected,y_train)

            
            y_pred=model_returned.predict(X_test_selected)
            class_labels = list(model_returned.classes_)
            model_training_obj=classification_report(y_test, y_pred, labels=class_labels)
            print(model_training_obj)
            print("Best model is",model_returned)

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_training_obj
               )

            return (
                 
                 self.model_trainer_config.trained_model_file_path,
            )





            
            

            
            
        except Exception as e:
            raise CustomException(e,sys)