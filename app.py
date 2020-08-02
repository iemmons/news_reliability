from flask import Flask
from pickle import load
from model import ReliabilityClassifier

app = Flask(__name__)

@app.route('/predict')
def predict():
    return "Hello World!"


if __name__ == '__main__':
    app.run()
