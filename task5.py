import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(grad, start_point, step_func):
    eps = 1e-5
    epoch = 100
    point = np.array(start_point)

    points = np.array([point])

    for i in range(1, epoch):
        vector = grad(point)
        next_point = point - step_func(i) * np.array(vector)
        distance = np.linalg.norm(point - next_point)

        if distance < eps:
            return points

        point = next_point
        points = np.append(points, [point], axis=0)
    return points


exp_step_func = (lambda iter: 1 / (1 + iter // 2))

f1 = (lambda xs: (xs[0] + 2) ** 2 + 1.5 * (xs[1] - 2) ** 2)
grad1 = (lambda xs: np.array([2 * (xs[0] + 2), 3 * (xs[1] - 2)]))

f2 = (lambda xs: 4 * (xs[0] / 2 + 2) ** 2 + 2 * (xs[1] / 4 - 2) ** 2)
grad2 = (lambda xs: np.array([4 * (xs[0] / 2 + 2), 1 * (xs[1] / 4 - 2)]))

fig, axs = plt.subplots(2)
fig.suptitle('Task 5')
tX = np.linspace(-6, 6, 1000)
tY = np.linspace(-2, 10, 1000)
X, Y = np.meshgrid(tX, tY)

points1 = gradient_descent(grad1, [5, 5], exp_step_func)
axs[0].plot(points1[:, 0], points1[:, 1], 'o-')
axs[0].contour(X, Y, f1([X, Y]), levels=sorted([f1(p) for p in points1]))

points2 = gradient_descent(grad2, [5, 5], exp_step_func)
axs[1].plot(points2[:, 0], points2[:, 1], 'o-')
axs[1].contour(X, Y, f2([X, Y]), levels=sorted([f2(p) for p in points2]))

plt.savefig('task5.png')
