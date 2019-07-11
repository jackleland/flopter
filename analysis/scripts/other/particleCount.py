import numpy as np
import matplotlib.pyplot as plt
import sys

file_name = "datad2d.dat"
if sys.argv[1]:
    file_name = sys.argv[1]

A, B = np.loadtxt(file_name, delimiter=", ", unpack=True)

plt.plot(B[1:])
plt.show()
