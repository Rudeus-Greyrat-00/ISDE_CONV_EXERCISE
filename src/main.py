import pandas
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from conv_1d_kernels import CConvKernelTriangle, CConvKernelMovingAverage, CConvolutionalCombo


def plot_digit(image, shape=(28, 28)):
    plt.imshow(np.reshape(image, newshape=shape),
               cmap='gray')  # this would be the first image (each row is an image 28 by 28 rowed)


def plot_digits(x, title=None):  # plot a list of digit (x must be a list of np.array)
    plt.figure()
    if title is not None:
        plt.suptitle(title)
    for i in range(len(x)):
        plt.subplot(2, 5, i + 1)  # i + 1 because subplots start from 1
        plot_digit(x[i])  # call our plot image function
    if title is None:
        title = "unnamed_image"
    plt.savefig("../out/" + title + ".pdf")
    # plt.show()


data = pandas.read_csv("../data/mnist_train_small.csv")
data = np.array(data)

labels = data[:, 0]
images = data[:, 1:] / 255

kernel_size = 5

triangle = CConvKernelTriangle(kernel_size)
average = CConvKernelMovingAverage(kernel_size)

filter_list = [triangle, average, triangle]

filters = CConvolutionalCombo(filter_list)

random_unique_digit = []
for cls in np.sort(np.unique(labels)):
    cls_images = images[labels == cls, :]
    rand_index = random.randint(0, cls_images.shape[0])
    random_unique_digit.append(cls_images[rand_index, :])

plot_digits(random_unique_digit, "Unperturbed images")

perturbed_average = []
perturbed_triangle = []
perturbed_combo = []
for digit in random_unique_digit:
    perturbed_triangle.append(triangle.kernel(digit))
    perturbed_average.append(average.kernel(digit))
    perturbed_combo.append(filters.kernel(digit))

plot_digits(perturbed_average, "Perturbed average")
plot_digits(perturbed_triangle, "Perturbed triangle")
plot_digits(perturbed_combo, "Perturbed combo")
