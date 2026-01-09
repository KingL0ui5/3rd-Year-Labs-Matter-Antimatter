"""
Task 1 09/01
Plot invariant mass histograms for 2011 dataset 
"""

import pickle
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")

with open('data/dataset_2011.pkl', 'rb') as infile:
    data_2011 = pickle.load(infile)

plt.hist(data_2011['B invariant mass'], bins=100)
plt.xlabel(r'B candidate mass / MeV/$c^2$')
plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
plt.show()

plt.hist(data_2011[abs(data_2011['dimuon-system invariant mass'] -
         3097) > 100]['B invariant mass'], bins=100)
plt.xlabel(r'B candidate mass / MeV/$c^2$')
plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
plt.show()
