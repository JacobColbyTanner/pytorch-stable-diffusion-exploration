
import torch
import numpy as np
import matplotlib.pyplot as plt 



timesteps = torch.abs(torch.randn(10000)) * (1000 / 4)
timesteps = timesteps.long().clamp(0, 1000 - 1)

#plot historgam of timesteps
plt.hist(timesteps.cpu().numpy(), bins=100)
plt.show()