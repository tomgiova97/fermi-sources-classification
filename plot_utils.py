import matplotlib.pyplot as plt
import numpy as np


def create_puls_agn_probs_histogram(puls_probs, agn_probs):

    # I need only the probability of being a pulsar from each source
    puls_probs = [prob[0] for prob in puls_probs]
    agn_probs = [prob[0] for prob in agn_probs]

    bins = np.linspace(0, 1, 21)

    plt.figure(1)
    plt.xlabel("Probability of being a pulsar")
    plt.ylabel("Count density")
    plt.title("Distribution of probability predictions for pulsar-AGN")
    plt.hist(puls_probs, bins=bins, color="yellow", label="psr", density="true")
    plt.hist(agn_probs, bins=bins, color="red", label="agn", density="true")
    plt.legend(loc="upper center")
    plt.xlim((0, 1))
    plt.show()


def create_unassociates_sources_histogram(unass_probs):
    # I need only the probability of being a pulsar from each source
    unass_probs = [prob[2] for prob in unass_probs]

    bins = np.linspace(0, 1, 21)

    plt.figure(1)
    plt.xlabel("Probability of being a 'Other'")
    plt.ylabel("Count density")
    plt.title("Distribution of probability predictions for Unassociated Sources")
    plt.hist(
        unass_probs, bins=bins, color="green", label="Other sources", density="true"
    )
    plt.legend(loc="upper center")
    plt.xlim((0, 1))
    plt.show()
