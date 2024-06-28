from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/change_text')
def change_text():
    new_text = "New Text"
    return new_text

@app.route('/predict_data', methods=['POST'])
def predict_data():
    json_data = request.get_json()
    data_list = []
    
    data_list.append(json_data)
    knn_from_joblib = joblib.load('price_prediction.pkl') 
    X_test = pd.DataFrame(data_list,columns=['Car','Total Mass','Painted Body Colour','Painted Gloss Black','Grained plastic','Job 1 Date','Country of production','Vehicle Volume','Number of parts within construction'])
    mapp =  {
    'UK':0,
    'SVK':1
    }
    X_test['Country of production'] = X_test['Country of production'].map(mapp)
    X_test['Country of production'] = X_test['Country of production'].astype(int)
    knn_from_joblib.predict(X_test) 
    knn_from_joblib = joblib.load('price_prediction.pkl') 
    y_pred = knn_from_joblib.predict(X_test) 

    # print(json_data) 
    return f'Predicted Cost is {y_pred[0]} Euros'  

app.run()
