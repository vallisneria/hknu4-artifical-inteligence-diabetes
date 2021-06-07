from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import googleapiclient.discovery
import os


np.random.seed(42)
app = Flask(__name__)
app.config["SECRET_KEY"] = "VpYGWgzSNUA5m9zQYEukMxKGuWh3wvAQuwj"
Bootstrap(app)


class LabForm(FlaskForm):
    preg = StringField("# Pregnancies", validators=[DataRequired()])
    glucose = StringField("Glucode", validators=[DataRequired()])
    blood = StringField("Blood pressure", validators=[DataRequired()])
    skin = StringField("Skin thickness", validators=[DataRequired()])
    insulin = StringField("Insulin", validators=[DataRequired()])
    bmi = StringField("BMI", validators=[DataRequired()])
    dpf = StringField("DPF Score", validators=[DataRequired()])
    age = StringField("Age", validators=[DataRequired()])
    submit = SubmitField("Submit")


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=["GET", "POST"])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = np.array([[
            float(form.preg.data),
            float(form.glucose.data),
            float(form.blood.data),
            float(form.skin.data),
            float(form.insulin.data),
            float(form.bmi.data),
            float(form.dpf.data),
            float(form.age.data)
        ]])

        data = pd.read_csv("./diabetes.csv", sep=",")
        X = data.values[:, 0:8]
        y = data.values[:, 8]

        scaler = MinMaxScaler()
        scaler.fit(X)

        X_test = scaler.transform(X_test)

        MODEL_NAME = "my_pima_model"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
        project_id = ""
        model_id = MODEL_NAME
        model_path = "projects/{}/models/{}".format(project_id, model_id)
        model_path += "/version/v0001"
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

        input_data_json = {"signature_name": "serving_default", "instances": X_test.tolist()}
        request = ml_resource.predict(name=model_path, body=input_data_json)
        response = request.execute()

        if "error" in response:
            raise RuntimeError(response["error"])

        predD = np.array([pred["dense_2"] for pred in response["predictions"]])
        res = predD[0][0]

        return render_template("result.html", res=res)
    return render_template("prediction.html", form=form)


if __name__ == '__main__':
    app.run(port=5000)
