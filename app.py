from flask import Flask ,request,url_for,render_template,redirect
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('home.html')


@app.route('/project')
def project():
    return render_template('index.html')


@app.route('/predict',methods=["POST","GET"])
def prediction():
    prediction_value = -1
    if request.method == "POST":
        company = str(request.form['company'])
        typename = str(request.form['typename'])
   
        inches = float(request.form['inches'])
        ss = str(request.form['screenresolution'])
        screenresolution = float(ss.split('x')[1]) * float(ss.split('x')[0])
        cpu = str(request.form['cpu'])
        ram = int(request.form['ram'])
        gpu = str(request.form['gpu'])
        os = str(request.form['os'])
        weight = float(request.form['weight'])
        memory = float(request.form['memory'])
        memorytype = str(request.form['memorytype'])
       
        # Load the OneHotEncoder and transform categorical features
        ohe = joblib.load('./models/oneHotEncoder.lb')
        converted_onehot = ohe.transform(pd.DataFrame([[company, typename, cpu, gpu, os, memorytype]])).toarray()
        
        # Convert numerical features to 2D arrays
        inches = np.array([[inches]])
        screenresolution = np.array([[screenresolution]])
        ram = np.array([[ram]])
        weight = np.array([[weight]])
        memory = np.array([[memory]])

        # Combine numerical and one-hot encoded categorical features
        combined = np.hstack((inches, screenresolution, ram, weight, memory, converted_onehot))

        # Load the model and make a prediction
        model_obj = joblib.load('./models/RandomForest.lb')
        prediction_value = model_obj.predict(combined)[0]  # Assuming model returns an array, take the first value

    return render_template('final.html', output=prediction_value)



         






if __name__ =="__main__":
    app.run(debug=True)

       
# encoded= np.hstack((df[["Inches","ScreenResolution","Ram","Weight","Memory_size","Price_rupee"]].values, temp))

    # ['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram',
    #    'Gpu', 'OpSys', 'Weight', 'Memory_size', 'Memory_type', 'Price_rupee']