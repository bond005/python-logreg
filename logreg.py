import math
import numpy


class LogRegError(Exception):
    def __init__(self, error_msg=None):
        self.msg = error_msg

    def __str__(self):
        error_msg = 'Logistic regression algorithm is incorrect!'
        if self.msg is not None:
            error_msg += (' ' + self.msg)
        return error_msg


class LogisticRegression:
    def __init__(self):
        self.__b = None
        self.__a = None
        self.__th = None

    def save(self, file_name):
        if (self.__a is None) or (self.__b is None) or (self.__th is None):
            raise LogRegError('Parameters have not been specified!')
        with open(file_name, 'w') as fp:
            fp.write('Input size {0}\n\n'.format(self.__a.shape[0]))
            for ind in range(self.__a.shape[0]):
                fp.write('{0}\n'.format(self.__a[ind]))
            fp.write('\n{0}\n\n{1}\n'.format(self.__b, self.__th))

    def load(self, file_name):
        with open(file_name, 'r') as fp:
            input_size = -1
            cur_line = fp.readline()
            ind = 0
            while len(cur_line) > 0:
                prepared_line = cur_line.strip()
                if len(prepared_line) > 0:
                    if input_size <= 0:
                        parts_of_line = prepared_line.split()
                        if len(parts_of_line) != 3:
                            raise LogRegError('Parameters cannot be loaded from a file!')
                        if (parts_of_line[0].lower() != 'input') or (parts_of_line[1].lower() != 'size'):
                            raise LogRegError('Parameters cannot be loaded from a file!')
                        input_size = int(parts_of_line[2])
                        if input_size <= 0:
                            raise LogRegError('Parameters cannot be loaded from a file!')
                        self.__a = numpy.zeros(shape=(input_size,), dtype=numpy.float)
                        self.__b = 0.0
                        self.__th = 0.5
                    else:
                        if ind > (input_size + 1):
                            raise LogRegError('Parameters cannot be loaded from a file!')
                        if ind < input_size:
                            self.__a[ind] = float(prepared_line)
                        elif ind == input_size:
                            self.__b = float(prepared_line)
                        else:
                            self.__th = float(prepared_line)
                            if (self.__th < 0.0) or (self.__th > 1.0):
                                raise LogRegError('Parameters cannot be loaded from a file!')
                        ind += 1
                cur_line = fp.readline()
            if ind <= (input_size + 1):
                raise LogRegError('Parameters cannot be loaded from a file!')

    def transform(self, X):
        if (self.__a is None) or (self.__b is None):
            raise LogRegError('Parameters have not been specified!')
        if (X is None) or (not isinstance(X, numpy.ndarray)) or (X.ndim != 2) or (X.shape[1] != self.__a.shape[0]):
            raise LogRegError('Input data are wrong!')
        return 1.0 / (1.0 + numpy.exp(-numpy.dot(X, self.__a) - self.__b))

    def predict(self, X):
        return (self.transform(X) >= self.__th).astype(numpy.float)

    def fit(self, X, y, eps=0.001, lr_min=0.0, lr_max=1.0, max_iters = 1000):
        if (X is None) or (y is None) or (not isinstance(X, numpy.ndarray)) or (X.ndim != 2) or\
                (not isinstance(y, numpy.ndarray)) or (y.ndim != 1) or (X.shape[0] != y.shape[0]):
            raise LogRegError('Train data are wrong!')
        if (eps <= 0.0) or (lr_min < 0.0) or (lr_max <= lr_min) or (max_iters < 1):
            raise LogRegError('Train parameters are wrong!')
        self.__a = numpy.random.rand(X.shape[1]) - 0.5
        self.__b = numpy.random.rand(1)[0] - 0.5
        f_old = self.__calculate_log_likelihood(X, y, self.__a, self.__b)
        print('{0:>5}\t{1:>17.12f}'.format(0, f_old))
        stop = False
        iterations_number = 1
        while not stop:
            gradient = self.__calculate_gradient(X, y)
            lr = self.__find_best_lr(X, y, gradient, lr_min, lr_max)
            self.__a = self.__a + lr * gradient[0]
            self.__b = self.__b + lr * gradient[1]
            f_new = self.__calculate_log_likelihood(X, y, self.__a, self.__b)
            print('{0:>5}\t{1:>17.12f}'.format(iterations_number, f_new))
            if (f_new - f_old) < eps:
                stop = True
            else:
                f_old = f_new
                iterations_number += 1
                if iterations_number >= max_iters:
                    stop = True
        if iterations_number < max_iters:
            print('The algorithm is stopped owing to very small changes of log-likelihood function.')
        else:
            print('The algorithm is stopped after the maximum number of iterations.')
        #self.__th = self.__calc_best_th(y, self.predict(X))
        self.__th = 0.5

    def __calculate_log_likelihood(self, X, y, a, b):
        eps = 0.000001
        p = 1.0 / (1.0 + numpy.exp(-numpy.dot(X, a) - b))
        return numpy.sum(y * numpy.log(p + eps) + (1.0 - y) * numpy.log(1.0 - p + eps))

    def __calculate_gradient(self, X, y):
        p = 1.0 / (1.0 + numpy.exp(-numpy.dot(X, self.__a) - self.__b))
        da = numpy.sum((y - p) * numpy.transpose(X), 1)
        db = numpy.sum(y - p)
        return (da, db)

    def __find_best_lr(self, X, y, gradient, lr_min, lr_max):
        theta = (1.0 + math.sqrt(5.0)) / 2.0
        eps = 0.00001 * (lr_max - lr_min)
        lr1 = lr_max - (lr_max - lr_min) / theta
        lr2 = lr_min + (lr_max - lr_min) / theta
        while abs(lr_min - lr_max) >= eps:
            y1 = self.__calculate_log_likelihood(X, y, self.__a + lr1 * gradient[0], self.__b + lr1 * gradient[1])
            y2 = self.__calculate_log_likelihood(X, y, self.__a + lr2 * gradient[0], self.__b + lr2 * gradient[1])
            if y1 <= y2:
                lr_min = lr1
                lr1 = lr2
                lr2 = lr_min + (lr_max - lr_min) / theta
            else:
                lr_max = lr2
                lr2 = lr1
                lr1 = lr_max - (lr_max - lr_min) / theta
        return (lr_max - lr_min) / 2.0

    def __calc_best_th(self, y_target, y_real):
        pass


def load_mnist_for_demo():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home='.')
    X_train = mnist.data[0:60000].astype(numpy.float) / 255.0
    y_train = mnist.target[0:60000]
    X_test = mnist.data[60000:].astype(numpy.float) / 255.0
    y_test = mnist.target[60000:]
    return ((X_train, y_train), (X_test, y_test))


if __name__ == '__main__':
    train_set, test_set = load_mnist_for_demo()
    classifiers = list()
    for recognized_class in range(10):
        new_classifier = LogisticRegression()
        new_classifier.fit(train_set[0], (train_set[1] == recognized_class).astype(numpy.float))
        classifiers.append(new_classifier)
    n_test_samples = test_set[0].shape[0]
    outputs = numpy.empty((n_test_samples, 10), dtype=numpy.float)
    for recognized_class in range(10):
        outputs[:, recognized_class] = classifiers[recognized_class].transform(test_set[0])
    results = outputs.argmax(1)
    n_errors = numpy.sum(results != test_set[1])
    print('Errors on test set: {0:%}'.format(float(n_errors) / float(n_test_samples)))