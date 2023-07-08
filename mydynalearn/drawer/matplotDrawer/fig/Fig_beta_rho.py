import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
from matplotlib.lines import Line2D
class Fig_beta_rho():
    def __init__(self):
        pass
    def plot(self, betaList,stady_rho_list):
        plt.plot(betaList,stady_rho_list)
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\rho$')
        plt.grid(True)