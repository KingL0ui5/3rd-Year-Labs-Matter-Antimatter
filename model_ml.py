# Apply the ML methods to the split datasets.

import xgboost
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

import histograms

# Take a look at the datasets and see how they separate
sig, bkg = histograms.data()
