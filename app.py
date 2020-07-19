# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:29:54 2020

@author: Shubh Gupta
"""

#importing libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#initializing flask app
app = Flask(__name__)
model = pickle.load(open('C://Users//Shubh Gupta//Desktop//DataScience//Github//loan predictor//loan-predictor.pkl', 'rb'))

#creating home function
@app.route('/')
def home():
    return render_template('index.html')

#creating predict function
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        ed = int(request.form['ed'])
        employ = int(request.form['employ'])
        address = int(request.form['address'])
        income = int(request.form['income'])
        debtinc = float(request.form['debtinc'])
        creddebt = float(request.form['creddebt'])
        othdebt = float(request.form['othdebt'])
        
        data = np.array([[age, ed, employ, address, income, debtinc, creddebt, othdebt]])
        model_prediction = model.predict(data)
        
        return render_template('result.html', prediction=model_prediction)
    
#initializing main function
if __name__ == "__main__":
    app.run(debug=True)

    