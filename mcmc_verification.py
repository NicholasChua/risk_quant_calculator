from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
import numpy as np

def verify_mcmc_implementation(control_effectiveness, num_simulations):
    plt.hist(control_effectiveness, bins=50, color='purple', alpha=0.7)
    plt.title('Distribution of Control Effectiveness from MCMC')
    plt.xlabel('Control Effectiveness')
    plt.ylabel('Frequency')
    plt.savefig('distribution_of_control_effectiveness.png')
    plt.show()

    plt.plot(control_effectiveness)
    plt.title('Trace Plot of Control Effectiveness')
    plt.xlabel('Iteration')
    plt.ylabel('Control Effectiveness')
    plt.savefig('trace_plot_of_control_effectiveness.png')
    plt.show()

    autocorrelation_plot(control_effectiveness)
    plt.title('Autocorrelation Plot of Control Effectiveness')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.savefig('autocorrelation_plot.png')
    plt.show()

    acceptance_rate = len(np.unique(control_effectiveness)) / num_simulations
    print(f"Acceptance Rate: {acceptance_rate * 100:.2f}%")
