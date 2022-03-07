import matplotlib.pyplot as plt
import numpy as np


def f(xs):
    return 5 * np.log(1 + xs[0] ** 2) + 10 * np.sin(xs[1] / 10)


def grad(xs):
    return np.array([10 * xs[0] / (1 + xs[0] ** 2), np.cos(xs[1] / 10)])


def gradient_descent(learning_rate):
    eps = 1e-5
    epoch = 100
    point = np.array([-5, 0])

    points = np.zeros((epoch, 2))
    points[0] = point

    for i in range(1, epoch):
        next_point = point - learning_rate * np.array(grad(point))
        distance = np.linalg.norm(point - next_point)

        if distance < eps:
            return i, point, points

        points[i] = point = next_point
    return epoch, point, points


learning_rate_iterations = np.arange(0.001, 2, 0.001)  # Массив, в котором хранятся скорости обучения
final_values = []
final_epochs = []

for learning_rate in learning_rate_iterations:
    epochs, point, _ = gradient_descent(learning_rate)
    final_values.append(f(point))  # Добавляем значения функции от конечной точки за опеределенное кол-во эпох
    final_epochs.append(epochs)  # Добавляем кол-во эпох для градиентного спуска, за которое функция сходится

# Отрисовка графиков
fig, axs = plt.subplots(3)
fig.suptitle('Task 1')
axs[0].set_xlabel("Learning rate")
axs[0].set_ylabel("Function value")
axs[0].plot(learning_rate_iterations, final_values)
axs[1].set_xlabel("Learning rate")
axs[1].set_ylabel("Epochs")
axs[1].plot(learning_rate_iterations, final_epochs)

# Best result:
epochs, point, points = gradient_descent(0.75)
tX = np.linspace(-10, 10, 1000)
tY = np.linspace(-20, 0, 1000)
X, Y = np.meshgrid(tX, tY)
axs[2].plot(points[:, 0], points[:, 1], 'o-')

axs[2].contour(X, Y, f([X, Y]), levels=sorted([f(p) for p in points]))
plt.savefig('task1.png')
