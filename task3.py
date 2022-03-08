import matplotlib.pyplot as plt
import numpy as np


def f(xs):
    return 5 * np.log(1 + xs[0] ** 2) + 10 * np.sin(xs[1] / 10)


def grad(xs):
    return np.array([10 * xs[0] / (1 + xs[0] ** 2), np.cos(xs[1] / 10)])


def golden_section(func, l, r):
    eps = 1e-1
    phi = (1 + np.sqrt(5)) / 2
    point_l = l
    point_r = r
    while point_r - point_l > eps:
        fraction = (point_r - point_l) / phi
        new_l = point_r - fraction
        new_r = point_l + fraction
        f_new_l = golden_section_borders.get(new_l, None)
        f_new_r = golden_section_borders.get(new_r, None)
        if not f_new_l:
            f_new_l = func(new_l)
            golden_section_borders[new_l] = f_new_l
        if not f_new_r:
            f_new_r = func(new_r)
            golden_section_borders[new_r] = f_new_r
        if f_new_l > f_new_r:
            point_l = new_l
        else:
            point_r = new_r
    return (point_l + point_r) / 2


def golden_section_step_func(_0, point, vector):
    def point_value(alpha):
        return f(point - alpha * vector)

    alpha = 1.0
    origin_value = point_value(0)
    alpha_value = point_value(alpha)
    if origin_value > alpha_value:
        for i in range(10):
            alpha *= 2
            next_alpha_value = point_value(alpha)
            if next_alpha_value > alpha_value:
                break
            alpha_value = next_alpha_value
    global golden_section_borders
    global func_calls_count
    func_calls_count = func_calls_count + len(golden_section_borders)
    golden_section_borders = {}
    return golden_section(point_value, 0.0, alpha)


def gradient_descent(step_func):
    epoch = 100
    point = np.array([-5, 0])
    points = []

    for i in range(1, epoch):
        vector = grad(point)
        next_point = point - step_func(i, point, vector) * np.array(vector)
        points.append([i, f(point)])
        point = next_point
    return points


golden_section_borders = {}
func_calls_count = 0

plots = []
plt.xlabel("Iterations count")
plt.ylabel("Function value")

for learning_rate, label in [
    [(lambda iter, _0, _1: 0.25), "const 0.25"],
    [golden_section_step_func, "golden section"]
]:
    points = gradient_descent(learning_rate)
    points = np.array(points).T
    plots.append(plt.plot(points[0], points[1], label=label)[0])

print("Function calls for const: " + str(99))
print("Function calls for golden section: " + str(func_calls_count))

plt.suptitle('Task3')
plt.legend(handles=plots)
plt.savefig('task3.png')


