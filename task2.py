import matplotlib.pyplot as plt
import numpy as np


def f(xs):
    return 5 * np.log(1 + xs[0] ** 2) + 10 * np.sin(xs[1] / 10)


def grad(xs):
    return np.array([10 * xs[0] / (1 + xs[0] ** 2), np.cos(xs[1] / 10)])


def gradient_descent(step_func):
    epoch = 100
    point = np.array([-5, 0])
    points = []

    for i in range(1, epoch):
        next_point = point - step_func(i) * np.array(grad(point))
        points.append([i, f(point)])
        point = next_point
    return points


plots = []
plt.xlabel("Iterations count")
plt.ylabel("Function value")

for learning_rate, label in [
    [(lambda iter: 0.25), "const 0.25"],
    [(lambda iter: 0.95 ** iter), "exp 0.95"],
    [(lambda iter: 1 / (1 + iter // 5)), "stair 1 5"]
]:
    points = gradient_descent(learning_rate)
    points = np.array(points).T
    plots.append(plt.plot(points[0], points[1], label=label)[0])

plt.suptitle('Task2')
plt.legend(handles=plots)
plt.savefig('task2.png')
