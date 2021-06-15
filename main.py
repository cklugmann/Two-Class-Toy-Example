import copy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm


def dist_to_hyperplane(x, y):
    d1 = 0.5 * (x - y)
    d2 = 0.5 * (y - x)
    s = 2 * (y >= x).astype(int) - 1
    d = s * np.sqrt(d1 ** 2 + d2 ** 2)
    return d


def sigmoid(x, alpha=1.):
    return 1. / (1. + np.exp(-1. * alpha * x))


def noisy_labels(x, y, alpha=1.):
    res = sigmoid(dist_to_hyperplane(x, y), alpha=alpha)
    return res


def cross_entropy(y, y_pred, eps=1e-6):
    return np.sum(-1. * y * np.log(y_pred + eps), axis=-1)


def batch_cross_entropy(y, y_pred):
    return np.mean(cross_entropy(y, y_pred))


def expand_labels(y):
    return np.concatenate([(1. - y).reshape(-1, 1), y.reshape(-1, 1)], axis=-1)


class Model:
    def __init__(self):
        self.logs = {'iteration': list(), 'train_loss': list(), 'val_loss': list()}

    def add_validation_set(self, X, y):
        self.X_val = X
        self.y_val = y

    def fit(self, X, y, lr=100., batch_size=64, n_epochs=1):
        """
            X: (N, D)
            y: (N,)
        """
        for k, log in self.logs.items():
            log.clear()
        D = X.shape[-1]
        self.params = np.concatenate([np.random.normal(size=(D,)), 0.5 + np.zeros((1,))])
        batches_per_epoch = X.shape[0] // batch_size
        it = 0
        for epoch in range(n_epochs):
            for batch in range(batches_per_epoch):
                indices = np.random.choice(X.shape[0], size=batch_size, replace=True)
                X_batch, y_batch = X[indices], y[indices]
                y_pred = self.predict(X_batch)
                loss = batch_cross_entropy(expand_labels(y_batch), expand_labels(y_pred))
                grads = self._grad(X_batch, y_batch, y_pred)
                if self.X_val is not None:
                    y_val_pred = self.predict(self.X_val)
                    val_loss = batch_cross_entropy(expand_labels(self.y_val), expand_labels(y_val_pred))
                self.params -= lr * grads
                output = 'Epoch: {} - batch {}/{}- loss: {:.3f}'.format(epoch+1, batch+1, batches_per_epoch, loss)
                self.logs['iteration'].append(it)
                self.logs['train_loss'].append(loss)
                if val_loss is not None:
                    output = ' - '.join([output, 'val loss: {}'.format(val_loss)])
                    self.logs['val_loss'].append(val_loss)
                it += 1
                print('\r' + output, end='')
        print('')

    def _util(self, X):
        X_augmented = np.concatenate([X, np.ones(X.shape[0]).reshape(-1,1)], axis=-1)
        return X_augmented, sigmoid(np.dot(X_augmented, self.params))

    def _grad(self, X, y, y_pred):
        X_augmented, activation = self._util(X)
        d_sigmoid = activation * (1. - activation)
        ratio = (1. - y) / (1. - y_pred + 1e-6) - y / (y_pred + 1e-6) 
        c = 1./X.shape[0] * ratio.reshape(1, -1) * d_sigmoid.reshape(1, -1)
        return np.mean(X_augmented.T * c, axis=-1)

    def predict(self, X):
        if self.params is not None:
            return self._util(X)[1]
        raise ValueError('Model not compiled.')


def generate_dataset(N, p_class=0.5, noisy=False, alpha=2):
    xs = np.random.uniform(low=0.0, high=1.0, size=(N,))

    indices = np.random.choice(N, size=int(p_class * N), replace=False)
    alt_indices = np.array([idx for idx in range(N) if not idx in indices])

    ys, labels = np.zeros_like(xs), np.zeros_like(xs, dtype=np.float32)

    ys[indices] = np.random.uniform(low=xs[indices], high=1. + np.zeros_like(indices))
    ys[alt_indices] = np.random.uniform(low=np.zeros_like(alt_indices), high=xs[alt_indices])

    if not noisy:
        labels[indices] = 1
    else:
        labels[indices] = noisy_labels(xs[indices], ys[indices], alpha=alpha)
        labels[alt_indices] = noisy_labels(xs[alt_indices], ys[alt_indices], alpha=alpha)

    X = np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1)], axis=-1)
    return X, labels


def main():
    N_train, N_test = 64, 32
    use_noisy_labels = True

    X_test, y_test = generate_dataset(N_test, noisy=use_noisy_labels)
    X_train, y_train = generate_dataset(N_train, noisy=use_noisy_labels)

    # Fit the model
    model = Model()
    model.add_validation_set(X_test, y_test)
    logs = list()
    repeats = 100
    test_accuracy = list()
    for _ in range(repeats):
        model.fit(X_train, y_train, lr=1., batch_size=16, n_epochs=32)
        current_logs = copy.deepcopy(model.logs)
        logs.append(current_logs)
        if not use_noisy_labels:
            # Prediction on test set
            y_pred = model.predict(X_test)
            test_accuracy.append(np.sum(((y_pred >= 0.5).astype(np.int32) == y_test).astype(np.int32)) / y_test.shape[0])

    # Compute loss statistics
    iterations = logs[-1]['iteration']
    losses = dict()
    for name in ['train_loss', 'val_loss']:
        loss = np.array([log[name] for log in logs])
        losses[name] = {
            'mean': np.mean(loss, axis=0),
            'std': np.std(loss, axis=0) / np.sqrt(repeats)
        }

    # Plot loss curves
    keys_to_labels = {
        'train_loss': 'train',
        'val_loss': 'val'
    }

    for k, v in losses.items():
        plt.plot(iterations, v['mean'], label=keys_to_labels[k])
        plt.fill_between(iterations, v['mean']-v['std'], v['mean']+v['std'], alpha=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    # Print the mean test accuracy if we consider the case of `hard` labels
    if not use_noisy_labels:
        print('Mean test accuracy: {:.3f} (± {:.3f})'.format(np.mean(test_accuracy), np.std(test_accuracy) / np.sqrt(repeats))) 

    # Decision boundary
    theta1, theta2, theta0 = list(model.params)
    xs = np.linspace(start=0., stop=1., num=100)
    ys = np.minimum(np.maximum(1./theta2 * ((0. - theta0) - theta1 * xs), 0), 1)

    # Grid
    n_grid = 1000
    _x_grid = np.linspace(start=0., stop=1., num=n_grid)
    _y_grid = np.linspace(start=0., stop=1., num=n_grid)
    x_grid, y_grid = np.meshgrid(_x_grid, _y_grid)
    if use_noisy_labels:
        label_grid = noisy_labels(x_grid, y_grid, alpha=2)
    else:
        label_grid = (y_grid > x_grid).astype(np.float32)

    # Prediction for every single grid point
    X_grid = np.concatenate([x_grid.reshape(-1,1), y_grid.reshape(-1, 1)], axis=-1)
    predicted = model.predict(X_grid).reshape(n_grid, n_grid)

    # Plot 2D filled contour
    cmap = sns.cubehelix_palette(as_cmap=True)
    f, ax = plt.subplots()
    ax.contourf(x_grid, y_grid, predicted, levels=32 if not use_noisy_labels else 32, cmap=cmap)
    if not use_noisy_labels:
        ax.plot(xs, ys, '-')
    points = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap=cmap, edgecolor='w')
    f.colorbar(points)
    plt.show()

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf = ax.plot_surface(x_grid, y_grid, label_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(0., 1.)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    main()