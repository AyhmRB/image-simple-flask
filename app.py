import os

from flask import Flask, request, render_template, send_from_directory, jsonify 
import joblib
import numpy as np

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#Additional Tests
@app.route("/callthis")
def test():
    return "<h1> hello there </h1>"

@app.route("/testjson")
def testjson(): 
    x = {
        "test":"value www",
        "test2":"valuessdsaf"
        }
    return jsonify(x)
#Additional Tests

@app.route("/ai", methods = ["POST"])
def predict():
    
    lr = request.args.get('data')
    lr = lr.split(',')
    lr = np.array(lr).astype('float64')
    
    model = joblib.load('ai_test.joblib')
    
    y = model.predict(np.array([lr]).reshape(-1,1))
    
    
    y = str(y)
    
    return y



if __name__ == "__main__":
    app.run()
