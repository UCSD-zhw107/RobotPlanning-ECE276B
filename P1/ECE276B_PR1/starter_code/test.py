import numpy as np



ORI = np.array([
            [1,0],
            [0,1],
            [0,-1],
            [-1,0]
        ])
x = np.asarray((1,3))
y = (1,3)
print(tuple(x) == y)