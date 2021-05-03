def tabgen (arr):
    rows, cols = arr.shape
    string = ''
    for i in range(rows):
        string += ' & '.join([str(j) for j in arr[i]])
        string += '\n\\\\\n'
    return string

import numpy as np

print(tabgen(np.zeros((4,3), dtype=int)))