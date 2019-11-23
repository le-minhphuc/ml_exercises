"""
Author: Le Minh Phuc
Date: 23 Nov 2019
"""

DIMENSION = 2 # for simplicity
SAMPLE_SIZE = 20 # for simplicity

def get_misclassified(w, sample):
	xtyt = np.zeros(3)
	for i in range(SAMPLE_SIZE):
		xiyi = sample[i]
		x_temp = np.array(xiyi, copy=True)
		x_temp[2] = -1
		if not np.sign(np.dot(w, x_temp)) == xiyi[2]:
			xtyt = xiyi
			return (True, xtyt)
	return (False, np.zeros(3))

# Generate linearly separable data
import numpy as np
ideal_line = np.random.rand(3) - 0.5
a, b, c = ideal_line # line ax + by = c
sample = np.zeros((20, 3))
for i in range(SAMPLE_SIZE):
	x_i = np.random.rand(3) - 0.5
	x_i[2] = -1
	y_i = 1 if np.dot(ideal_line, x_i) > 0 else -1
	sample[i] = np.array(x_i, copy=True)
	sample[i][2] = y_i
print("Sample:\n{}".format(sample))

# Run perceptron learning algorithm on sample
w = np.zeros(3) # initial weight vector starts at (0, 0, 0)
iterations_count = 0
while True:
	has_misclassified, data_t = get_misclassified(w, sample)
	if has_misclassified:
		# update w
		print("mis-classified point {}".format(data_t))
		x_t = np.array(data_t, copy=True)
		x_t[2] = -1
		y_t = data_t[2]
		w = w + y_t*x_t
		iterations_count += 1
	else:
		break

if get_misclassified(w, sample)[0]:
	print("Perceptron learning failed!")
else:
	print("Perceptron learning succeeded after {} iterations!".format(iterations_count))

import matplotlib.pyplot as plt
sample_blue = sample[sample[:, 2] == -1]
sample_green = sample[sample[:, 2] == 1]
plt.plot(sample_blue[:,0], sample_blue[:,1], 'bo')
plt.plot(sample_green[:,0], sample_green[:,1], 'go')

ideal_line_x_coords = np.arange(-0.5, 0.5, 0.0001)
ideal_line_y_coords = (c - a*ideal_line_x_coords)/b
plt.plot(ideal_line_x_coords, ideal_line_y_coords, 'black')

w_x_coords = np.arange(-0.5, 0.5, 0.0001)
w_y_coords = (w[2] - w[0]*w_x_coords)/w[1]
plt.plot(w_x_coords, w_y_coords, 'red')

plt.show()