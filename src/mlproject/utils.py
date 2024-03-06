
#utils for generic functionality

import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from dotenv import load_dotenv   ##loads all environment variables
import pymysql
import pickle 
import numpy
import pandas
from sklearn.metrics import accuracy_score
load_dotenv()

host=os.getenv('host')
user=os.getenv('root')
password=os.getenv('password')
db=os.getenv('db')
print("Database",db)
def read_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established",mydb)
        dframe=pd.read_sql_query('Select * from projects.dataset_full', mydb)
        print(dframe.head())


        return dframe
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train,X_test,y_test,model,param_dist):
    try:
        
        rscv = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        rscv.fit(X_train, y_train)

        model.set_params(**rscv.best_params_)
        best_model = rscv.best_estimator_

        best_params = rscv.best_params_
        print("Best Hyperparameters:", best_params)

        model.fit(X_train, y_train)
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_model_score = accuracy_score(y_train, y_train_pred)
        test_model_score = accuracy_score(y_test, y_test_pred)
        
        best_model_score = best_model.score(X_test, y_test)

        if best_model_score<0.72:
                raise CustomException("No best model found")
        logging.info(f"Best found model on both training and testing dataset") 
        
        return best_model
    
    except Exception as e:
        raise CustomException(e, sys)
      

        
    