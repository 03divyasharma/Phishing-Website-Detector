from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.mlproject.pipelines.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__) #entry point of the program
app=application
# creating specific app url
@app.route("/")   #parameters; two types of methods- get and post #parameters; two types of methods- get and post
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            qty_slash_url=request.form.get('qty_slash_url'),
            qty_underline_directory=request.form.get('qty_underline_directory'),
            file_length=request.form.get('file_length'),
            length_url=request.form.get('length_url'),
            time_domain_activation=request.form.get('time_domain_activation'),
            qty_dot_domain=request.form.get('qty_dot_domain'),
            asn_ip=request.form.get('asn_ip'),
            ttl_hostname=request.form.get('ttl_hotname'),
            time_domain_expiration=request.form.get('time_domain_expiration'),
            time_response=request.form.get('time_response'),
            qty_hyphen_url=request.form.get('qty_hyphen_url'),
            qty_percent_params=request.form.get('qty_percent_params'),
            qty_vowels_domain=request.form.get('qty_vowels_domain'),
            qty_hyphen_params=request.form.get('qty_hyphen_params'),
            qty_dot_url=request.form.get('qty_dot_url'),
            qty_equal_url=request.form.get('qty-equal_url'),
            qty_nameservers=request.form.get('qty_nameservers'),
            qty_mx_servers=request.form.get('qty_mx_servers'),
            qty_ip_resolved=request.form.get('qty_ip_resolved'),
            qty_slash_params=request.form.get('qty_slash_params'),
            qty_redirects=request.form.get('qty_redirects'),
            qty_tld_url=request.form.get('qty_tld_url'),
            tls_ssl_certificate=request.form.get('tls_ssl_certificate'),
            qty_underline_url=request.form.get('qty_underline'),
            domain_spf=request.form.get('domain_spf')
)

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        final_results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=final_results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")