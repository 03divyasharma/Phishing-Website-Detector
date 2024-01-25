
#utils for generic functionality

import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

import pandas as pd
from dotenv import load_dotenv   ##loads all environment variables
import pymysql
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


