
import os
import sys


from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import read_data
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #initial data

    def initiate_data_ingestion(self):
        
        try:
            ##reading code from mysql
            dframe=read_data()
            logging.info('Reading completed')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #

            dframe.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            
            train_set,test_set=train_test_split(dframe,test_size=0.2,random_state=42)

            dframe.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            dframe.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
        
