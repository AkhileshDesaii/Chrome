from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def result():
    ID = float(request.form['ID'])
    Thumbs_Up = float(request.form['Thumbs_Up'])

    X = np.array([[ID, Thumbs_Up]])
    model = pickle.load(open('model.pkl','rb'))
    y_predict = model.predict(X)
    return jsonify({'Prediction': float(y_predict)})

if __name__ == "__main__":
    app.run(debug=True,port=1234)
