import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path=os.path.join('artifacts',"preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=['qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
       'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
       'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url',
       'qty_comma_url', 'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url',
       'qty_dollar_url', 'qty_percent_url', 'qty_tld_url', 'length_url',
       'qty_dot_domain', 'qty_hyphen_domain', 'qty_underline_domain',
       'qty_at_domain', 'qty_vowels_domain', 'domain_length', 'domain_in_ip',
       'server_client_domain', 'qty_dot_directory', 'qty_hyphen_directory',
       'qty_underline_directory', 'qty_slash_directory',
       'qty_questionmark_directory', 'qty_equal_directory', 'qty_at_directory',
       'qty_and_directory', 'qty_exclamation_directory', 'qty_space_directory',
       'qty_tilde_directory', 'qty_comma_directory', 'qty_plus_directory',
       'qty_asterisk_directory', 'qty_hashtag_directory',
       'qty_dollar_directory', 'qty_percent_directory', 'directory_length',
       'qty_dot_file', 'qty_hyphen_file', 'qty_underline_file',
       'qty_slash_file', 'qty_questionmark_file', 'qty_equal_file',
       'qty_at_file', 'qty_and_file', 'qty_exclamation_file', 'qty_space_file',
       'qty_tilde_file', 'qty_comma_file', 'qty_plus_file',
       'qty_asterisk_file', 'qty_hashtag_file', 'qty_dollar_file',
       'qty_percent_file', 'file_length', 'qty_dot_params',
       'qty_hyphen_params', 'qty_underline_params', 'qty_slash_params',
       'qty_questionmark_params', 'qty_equal_params', 'qty_at_params',
       'qty_and_params', 'qty_exclamation_params', 'qty_space_params',
       'qty_tilde_params', 'qty_comma_params', 'qty_plus_params',
       'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params',
       'qty_percent_params', 'params_length', 'tld_present_params',
       'qty_params', 'email_in_url', 'time_response', 'domain_spf', 'asn_ip',
       'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved',
       'qty_nameservers', 'qty_mx_servers', 'ttl_hostname',
       'tls_ssl_certificate', 'qty_redirects', 'url_google_index',
       'domain_google_index', 'url_shortened']
            

            num_pipeline= Pipeline(steps=
                
                [("scaler", Normalizer())]
                )
            logging.info(f"Numerical Columns:{numerical_columns}")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            target_column=["phishing"]
            
            
            print(train_df.columns)
            print(test_df)
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            ## divide the test dataset to independent and dependent feature
            input_features_train_df=train_df.drop(columns=target_column,axis=1)
            target_feature_train_df=train_df["phishing"]

            input_feature_test_df=test_df.drop(columns=target_column,axis=1)
            target_feature_test_df=test_df["phishing"]

            logging.info("Applying Preprocessing on training and test dataframe")

            feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)
            
            train_arr = np.c_[
                feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[feature_test_arr],np.array(target_feature_test_df)

            #train_df_modified=pd.DataFrame(input_feature_train_arr,columns=train_df.columns)
            #test_df_modified=pd.DataFrame(input_feature_test_arr,columns=test_df)

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj=preprocessing_obj
               )

            return (
                 train_arr,
                 test_arr,
                 self.data_transformation_config.preprocessing_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)