import os

from flask import Flask, request, render_template, send_from_directory, jsonify 
import joblib
import numpy as np

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


#------------------------------------------Additional Tests
#Additional Tests------------------------------------------
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

@app.route("/aiform", methods = ["POST"])
def predict():
    lr = request.form.get('data')
    try:
        lr = lr.split(',')
    except: 
        return "<h1> Error in splitting </h1>"
    
    try:    
        lr = np.array(lr).astype('float64')
    except:
        return "<h1> error in coversion to float array </h1>"
    
    try:
        model = joblib.load('ai_test.joblib')
    except: 
        return "<h1> error in loading model </h1>"
    
    try:
        y = model.predict(np.array([lr]).reshape(-1,1))
    except:
        return "<h1> error in predicting </h1>"
    
    try:    
        y = str(y)
    except:
        return "<h1> error in converting to string </h1>"
    
    try:
        return y
    except:
        return '<h1> Error in returning Y </h1>'
    

#------------------------------------------Additional Tests
#Additional Tests------------------------------------------

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    return render_template("complete.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run()
    

