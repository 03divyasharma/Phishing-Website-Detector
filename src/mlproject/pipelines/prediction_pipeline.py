import sys,os
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):  #does the prediction
        try:
            model_path='artifacts\model.pkl'
            model=load_object(file_path=model_path) #loads the picke file
            preds=model.predict(features)
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:                                   #maps inputs which we are giving in html with the backend
    def __init__(self):
       pass
   

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "qty_slash_url":[self.qty_slash_url],              "qty_underline_directory":[self.qty_underline_directory],
                "file_length":[self.file_length],                  "length_url":[self.length_url],
                "time_domain_activation":[self.time_domain_activation], "qty_dot_domain":[self.qty_dot_domain],
                "asn_ip":[self.asn_ip],                                 "ttl_hostname":[self.ttl_hostname],
                "time_domain_expiratio":[self.time_domain_expiration],  "time_response":[self.time_response],
                "qty_hyphen_url":[self.qty_hyphen_url],                 "qty_percent_params":[self.qty_percent_params],
                "qty_vowels_domain":[self.qty_vowels_domain],           "qty_hyphen_params":[self.qty_hyphen_params],
                "qty_dot_url":[self.qty_dot_url],                       "qty_equal_url":[self.qty_equal_url],
                "qty_nameservers":[self.qty_nameservers],               "qty_mx_servers":[self.qty_mx_servers],
                "qty_ip_resolved":[self.qty_ip_resolved],               "qty_slash_params":[self.qty_slash_params],
                "qty_redirects":[self.qty_redirects],                   "qty_tld_url":[self.qty_tld_url],
                "tls_ssl_certificate":[self.tls_ssl_certificate],       "qty_underline_url":[self.qty_underline_url],
                "domain_spf":[self.domain_spf]  }

            

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)