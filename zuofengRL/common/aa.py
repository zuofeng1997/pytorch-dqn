import numpy as np
a = np.ones((4,84,84))
a = a.transpose(2,1,0)
print(a.shape)