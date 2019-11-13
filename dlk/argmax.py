import sys
import numpy as np
array=[]
for line in sys.stdin:
    if 'Output' in line or 'test' in line:
        continue
    array.extend([float(n) for n in line.strip('[] \r\n').split()])
print(array)
print(np.argmax(array))
