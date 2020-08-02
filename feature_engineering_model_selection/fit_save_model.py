"""
This is a convenience executable that will fit the model on a set of training data and save the resulting model
as a pickle file, to be used for later prediction. This helps the web application start faster and eliminates
the need to package the training data.
"""

from model import ReliabilityClassifier
from pickle import dump
import pandas as pd

model = ReliabilityClassifier()

data = pd.read_csv("../data/train.csv")
data = data[:100]
model.fit(data)

with open('../model.pickle', 'wb') as f:
    dump(model, f)
