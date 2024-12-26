import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset

def general_mc_data(nodes):
    X = np.random.rand(nodes ** 2).reshape(nodes, nodes)
    X = np.triu(X)
    X += X.T - np.diag(X.diagonal())
    for i in range(nodes):
        X[i,i]=0
    print(X)

general_mc_data(5)