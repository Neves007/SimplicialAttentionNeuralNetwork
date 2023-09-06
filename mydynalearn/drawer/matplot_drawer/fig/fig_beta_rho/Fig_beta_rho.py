import matplotlib.pyplot as plt
class Fig_beta_rho():
    def __init__(self):
        pass
    def plot(self, BETA_LIST,stady_rho_list):
        plt.plot(BETA_LIST,stady_rho_list)
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\rho$')
        plt.ylim(0,0.8)
        plt.grid(True)