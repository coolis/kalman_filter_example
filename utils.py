import os
import matplotlib.pyplot as plt

baseDir = os.path.dirname(__file__)

def line(x, y, xlabel, ylabel, title):
    fig = plt.figure()
    plt.plot(x,y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    fig.savefig(os.path.join(baseDir, 'Figures/'+title+'.png'))
