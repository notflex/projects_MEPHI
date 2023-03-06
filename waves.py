import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def initial_conditions1():
    u0 = lambda x, t: t + x ** 2 + np.arcsin(t * (x - 1) / 2)  # exact solution
    phi = lambda x: x ** 2  # first initial condition, u(x, 0)
    phi2 = lambda x: 2  # second derivative of phi(x)
    psi = lambda x: (x + 1) / 2  # second initial condition, u_t(x, 0)
    f = lambda x, t: -1 + ((t * ((x - 1) ** 3) - (t ** 3) * (x - 1)) / ((4 - (t ** 2) * ((x - 1) ** 2)) ** (3 / 2)))
    alpha = np.array([1, 1])
    beta = np.array([0, 2])
    gamma = np.array([lambda t: t - np.arcsin(t / 2), lambda t: 5 + 2 * t])
    return u0, phi, phi2, psi, f, alpha, beta, gamma


def initial_conditions2():
    u0 = lambda x, t: 0  # exact solution
    phi = lambda x: 0  # first initial condition, u(x, 0)
    phi2 = lambda x: 0  # second derivative of phi(x)
    psi = lambda x:0  # second initial condition, u_t(x, 0)
    f = lambda x, t: 0
    alpha = np.array([1, 1])
    beta = np.array([0, 0])
    gamma = np.array([lambda t: 1*np.sin(10*t) if (t < np.pi/10) else 0, lambda t: 2*np.sin(10*t) if (t < np.pi/10) else 0])
    return u0, phi, phi2, psi, f, alpha, beta, gamma


def initial_conditions3():
    u0 = lambda x, t: 0 # exact solution
    phi = lambda x: 0  # first initial condition, u(x, 0)
    phi2 = lambda x: 0  # second derivative of phi(x)
    psi = lambda x: 0  # second initial condition, u_t(x, 0)
    f = lambda x, t: 0
    alpha = np.array([1, 1])
    beta = np.array([0, 0])
    gamma = np.array([lambda t: 1 if (t < np.pi / 10) else 0,
                      lambda t: 2 if (t < 0) else 0])
    return u0, phi, phi2, psi, f, alpha, beta, gamma


def initial_conditions4():
    u0 = lambda x, t: 0  # exact solution
    phi = lambda x: np.array(list(map(lambda z: 3*z if (z < 0.5) else 3. - 3 * z, x)))  # first initial condition, u(x, 0)
    phi2 = lambda x: 0  # second derivative of phi(x)
    psi = lambda x: 0  # second initial condition, u_t(x, 0)
    f = lambda x, t: 0
    alpha = np.array([1, 1])
    beta = np.array([0, 0])
    gamma = np.array([lambda t: 0,
                      lambda t: 0])
    return u0, phi, phi2, psi, f, alpha, beta, gamma


def initial_conditions5():
    u0 = lambda x, t: 0  # exact solution
    phi = lambda x: 2*np.sin(2*np.pi*x/(1/3))  # first initial condition, u(x, 0)
    phi2 = lambda x: 0  # second derivative of phi(x)
    psi = lambda x: 0  # second initial condition, u_t(x, 0)
    # u_tt = a**2 * u_xx + f(x, t)
    f = lambda x, t: 0
    alpha = np.array([1, 1])
    beta = np.array([0, 0])
    gamma = np.array([lambda t: 0,
                      lambda t: 0])
    return u0, phi, phi2, psi, f, alpha, beta, gamma


def least_squares(x, y):
    n = len(x)

    sumx = x.sum()
    sumy = y.sum()
    xy = x * y
    sumxy = xy.sum()
    xx = x * x
    sumxx = xx.sum()

    b = (n * sumxy - sumx*sumy) / (n * sumxx - sumx**2)
    a = (sumy - b * sumx) / n
    return a, b


def next_layer_first_order(u_prev, f,  alpha, beta, gamma, a,  h, tau, t_now):
    u = np.zeros(len(u_prev[0]))
    for i in range(1, len(u) - 1):
        u[i] = 2 * u_prev[1, i] - u_prev[0, i] + ((a * tau / h)**2) * (u_prev[1, i + 1] - 2 * u_prev[1, i] + u_prev[1, i - 1]) + tau**2 * f(i * h, t_now)
    u[0] = (gamma[0](t_now) - beta[0] * u[1] / h) / (alpha[0] - beta[0] / h)
    print(u[0])
    u[-1] = (gamma[1](t_now) + beta[1] * u[-2] / h) / (alpha[1] + beta[1] / h)
    return u


def next_layer_second_order(u_prev, f,  alpha, beta, gamma, a,  h, tau, t_now):
    u = np.zeros(len(u_prev[0]))
    c = (tau * a / h) ** 2
    for i in range(1, len(u) - 1):
        u[i] = 2 * u_prev[1, i] - u_prev[0, i] + c * (u_prev[1, i + 1] - 2 * u_prev[1, i] + u_prev[1, i - 1]) + tau**2 * f(i * h, t_now)
    if beta[0] == 0:
        u[0] = gamma[0](t_now) / alpha[0]
    else:
        u[0] = 2 * c * (u_prev[1, 1] + u_prev[1, 0] * (h * alpha[0] / beta[0] - 1) - h * gamma[0](t_now) / beta[
            0]) + tau ** 2 * f(0., t_now) + 2 * u_prev[1, 0] - u_prev[0, 0]
    if beta[1] == 0:
        u[-1] = gamma[1](t_now) / alpha[1]
    else:
        u[-1] = 2 * c * (
                    u_prev[1, -2] - u_prev[1, -1] * (h * alpha[1] / beta[1] + 1) + h * gamma[1](t_now) / beta[
                1]) + tau ** 2 * f(1., t_now) + 2 * u_prev[1, -1] - u_prev[0, -1]
    return u


def first_layer_first_order(u, tau, psi, x_range, phi2, f, a):
    return u + tau * psi(x_range)


def first_layer_second_order(u, tau, psi, x_range, phi2, f, a):
    return u + tau * psi(x_range) + ((tau**2)/2) * (a**2 * phi2(x_range) + f(x_range, 0))


def animation():
    x_min = 0.
    x_max = 1.
    t_min = 0.
    t_max = 40.
    h = 0.01
    N = int((x_max - x_min) // h)  # number of points
    x_range = np.linspace(x_min, x_max, N)
    C = 1.
    a = np.sqrt(1/2)
    tau = C * h / a
    first_layer = first_layer_second_order
    next_layer = next_layer_second_order

    u0, phi, phi2, psi, f, alpha, beta, gamma = initial_conditions2()

    u = np.zeros((3, N))
    u[0] = phi(x_range)
    u[1] = first_layer(u[0], tau, psi, x_range, phi2, f, a)
    u[2] = next_layer(u[0:2], f, alpha, beta, gamma, a, h, tau, t_min + 2 * tau)

    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim=(-2, 2))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = x_range
        t_now = t_min + i * tau
        if i in range(3):
            y = u[i]
        else:
            u[0] = u[1]
            u[1] = u[2]
            u[2] = next_layer(u[0:2], f, alpha, beta, gamma, a, h, tau, t_now)
            y = u[2]
        line.set_data(x, y)
        print(t_now)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=int((t_max - t_min) // tau), interval=20, blit=True)

    plt.show()
    anim.save('solution.gif', writer='imagemagick')


def order_of_approximation():
    x_min = 0.
    x_max = 1.
    t_min = 0.
    Nt = 25
    h_range = np.array([0.00025, 0.0005, 0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05])
    C = 1.
    a = np.sqrt(1 / 2)
    first_layer = first_layer_second_order
    next_layer = next_layer_second_order

    u0, phi, phi2, psi, f, alpha, beta, gamma = initial_conditions1()

    error = np.zeros(len(h_range))

    for h, i in zip(h_range, range(len(h_range))):
        print(h)
        N = int((x_max - x_min) // h)  # number of points
        x_range = np.linspace(x_min, x_max, N)
        tau = C * h / a
        t_max = t_min + tau * Nt
        t_range = np.linspace(t_min + 3 * tau, t_max, Nt - 2)
        u = np.zeros((3, N))
        u[0] = phi(x_range)
        u[1] = first_layer(u[0], tau, psi, x_range, phi2, f, a)
        u[2] = next_layer(u[0:2], f, alpha, beta, gamma, a, h, tau, t_min + 2 * tau)
        for t in t_range:
            u[0] = u[1]
            u[1] = u[2]
            u[2] = next_layer(u[0:2], f, alpha, beta, gamma, a, h, tau, t)
            error[i] = max(error[i], np.max(np.abs(u[2] - u0(x_range, t))))

    #h_range = np.log10(h_range)
    #error = np.log10(error)

    plt.suptitle('Зависимость логарифма абсолютной погрешности от логарифма шага интегрирования')
    plt.subplot(1, 1, 1)
    plt.xlabel("log(h)")
    plt.ylabel("log(max(|Δu|))")
    plt.grid()
    plt.loglog(h_range, error, color='k')

    coeffs = least_squares(np.log10(h_range), np.log10(error))
    print("linear regression", ": y(x) = ", coeffs[0], " + ", coeffs[1], "x", sep="")

    plt.show()


if __name__ == '__main__':
    order_of_approximation()
    animation()
