import pickle
from flask import Flask, render_template, request
application= Flask(__name__)
model=pickle.load(open('artifacts/model.pkl', 'rb'))

@application.route('/')

def index():
    return render_template('index.html')

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction= model.predict([[request.form.get('qty_slash_url')],
                                [request.form.get('qty_underline_directory')],
                                [request.form.get('file_length')],
                                [request.form.get('length_url')],
                                [request.form.get('time_domain_activation')],
                                [request.form.get('qty_dot_domain')],
                                [request.form.get('asn_ip')],
                                [request.form.get('ttl_hostname')],
                                [request.form.get('time_domain_expiration')],
                                [request.form.get('time_response')],
                                [request.form.get('qty_hyphen_url')],
                                [request.form.get('qty_percent_params')],
                                [request.form.get('qty_vowels_domain')],
                                [request.form.get('qty_hyphen_params')],
                                [request.form.get('qty_dot_url')],
                                [request.form.get('qty_equal_url')],
                                [request.form.get('qty_nameservers')],
                                [request.form.get('qty_mx_servers')],
                                [request.form.get('qty_ip_resolved')],
                                [request.form.get('qty_slash_params')],
                                [request.form.get('qty_redirects')],
                                [request.form.get('qty_tld_url')],
                                [request.form.get('tls_ssl_certification')],
                                [request.form.get('qty_underline_url')],
                                [request.form.get('domain_spf')],
                               
                               ])
    print(prediction)
    return render_template('index.html',prediction_text=f'The website is {prediction}')

if __name__=='__main__':
    application.run(debug=True)
