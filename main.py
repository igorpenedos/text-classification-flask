from model.model import Model
from flask import Flask, request
from http import HTTPStatus

model = Model()

model.init()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def accuracy():
    return {"accuracy": str(model.accuracy)}, HTTPStatus.OK

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return "Missing JSON Body", HTTPStatus.BAD_REQUEST
    params = request.get_json()
    if not len(params) or not "text" in params:
        return "Text param not provided", HTTPStatus.BAD_REQUEST
    text = params["text"]
    
    return {"prediction": model.predict(text)}, HTTPStatus.OK
