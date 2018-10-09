import numpy as np
import matplotlib.pyplot as plt

import stimulus

s = stimulus.MultiStimulus()


_, stim, hat, mk, _ = s.generate_trial(0)

for b in range(16):
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(stim[:,b,:], aspect='auto')
    ax[1].imshow(hat[:,b,:], aspect='auto')
    ax[2].imshow(mk[:,b,np.newaxis], aspect='auto')

    plt.show()
