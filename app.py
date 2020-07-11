import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('E:\project\diabetes.pkl', 'rb'))
@app.route('/dhruv')
@app.route('/')
def home():
    return render_template('project.html')

@app.route('/predict',methods=['POST'])
def predict():
    data1 = request.form['p']
    data2 = request.form['g']
    data3 = request.form['bp']
    data4 = request.form['s']
    data5 = request.form['i']
    data6 = request.form['b']
    data7 = request.form['d']
    data8 = request.form['a']
    predi = []
    for i in range(0,3):
        arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8]])
        pred = model.predict(arr)
        predi.append(pred)

    co = predi.count(1)
    if co == 2 or co== 3 :
        fin = 1
    else:
        fin=0
    return render_template('response.html', prediction_text=fin)


if __name__ == "__main__":
    app.run(debug=True)