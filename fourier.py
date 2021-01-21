import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(123)


class FourierBasisRegression(LinearRegression):

    def __init__(self, p=10, T=5, **kwargs):
        self.p = p
        self.T = T
        super().__init__(**kwargs)


    def fit(self, X, y):
        return super().fit(self._fourier_basis(X), y)


    def predict(self, X):
        return super().predict(self._fourier_basis(X))


    def _fourier_basis(self, x):

        X_d = np.zeros((len(np.atleast_1d(x)), self.p))
        X_d[:, 0] = 1.
        dw = 0.5 * (np.pi / self.T)

        for j in range(1, self.p):
            w = np.floor((j + 1.0001) / 2.) * dw

            if j % 2:
                X_d[:, j] = np.cos(w * x)
            else:
                X_d[:, j] = np.sin(w * x)

        return X_d


def f(x):
    return 2*np.sin(3*x) - np.cos(5*x) + x**2


def get_data(n_points, f=f):
    x = np.random.uniform(-2, 2, n_points)
    y =  f(x)

    return x, y




if __name__ == "__main__":
    x, y = get_data(20)
    x_ = np.linspace(x.min(), x.max(), 100)

    f = FourierBasisRegression(p=5)
    f.fit(x, y)

    plt.plot(x, y, '.')
    plt.plot(x_, f.predict(x_))
    plt.show()
