import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.5, num_iterations=2000, print_cost=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.w = []
        self.b = []

    def fit(self, X_train, y_train):
        # Initialize parameters with zeros
        w, b = self.initialize_with_zeros(X_train.shape[0])

        # Gradient descent
        parameters, grads, costs = self.optimize(w, b, X_train, y_train)

        self.w = parameters['w']
        self.b = parameters['b']

    def initialize_with_zeros(self, dim):
        w = np.zeros((dim, 1))
        b = 0

        return w, b

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))

        return s

    def propagate(self, w, b, X, y):
        m = X.shape[1]

        # Forward propagation (from X to cost)
        A = self.sigmoid(np.dot(w.T, X) + b)
        cost = - (1 / m) * np.sum((y * np.log(A) + (1 - y) * np.log(1 - A)))

        # Backward propagation (to find grad)
        dw = (1 / m) * np.dot(X, (A - y).T)
        db = (1 / m) * np.sum(A - y)

        cost = np.squeeze(cost)

        grads = {'dw': dw, 'db': db}

        return grads, cost

    def optimize(self, w, b, X, y):
        costs = []

        for i in range(self.num_iterations):
            # Costs and gradient calculation
            grads, cost = self.propagate(w, b, X, y)

            # Retrieve derivatives from grads
            dw = grads['dw']
            db = grads['db']

            # Update rule
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training iterations
            if self.print_cost and i % 100 == 0:
                print(f'Cost after iteration {i}: {cost}')

        params = {'w': w, 'b': b}
        grads = {'dw': dw, 'db': db}

        return params, grads, costs

    def predict(self, X):
        m = X.shape[1]
        y_prediction = np.zeros((1, m))
        self.w = self.w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)

        # Convert probabilities A[0, i] to actual predictions p[0, i]
        for i in range(A.shape[1]):
            y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

        return y_prediction
