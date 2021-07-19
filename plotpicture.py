import matplotlib.pyplot as plt
import numpy as np
def plot(a,b,c,d):
    figure=plt.figure()
    ax1=figure.add_subplot(4,1,1)
    ax2=figure.add_subplot(4,1,2)
    ax3=figure.add_subplot(4,2,1)
    ax4=figure.add_subplot(4,2,2)
    ax1.plot(np.arange(0,len(a)),a)
    ax2.plot(np.arange(0, len(b)), b)
    ax3.plot(np.arange(0, len(c)), c)
    ax4.plot(np.arange(0, len(d)), d)
    plt.show()
