import numpy as np
from sklearn.linear_model import Ridge

class ESN:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=1.25, sparsity=0.2, alpha=0.9):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.alpha = alpha

        # Initialize reservoir weights
        self.Win = np.random.rand(self.reservoir_size, self.input_size) * 2 - 1
        self.W = np.random.rand(self.reservoir_size, self.reservoir_size) * 2 - 1
        self.W[np.random.rand(*self.W.shape) > self.sparsity] = 0

        # Scale the reservoir matrix to ensure the echo state property
        eigenvalues, _ = np.linalg.eig(self.W)
        self.W *= self.spectral_radius / np.max(np.abs(eigenvalues))

        # Output weights (learned during training)
        self.Wout = None

    def _update_reservoir(self, x, prev_state):
        # Update the reservoir state with leaky integration
        return (1 - self.alpha) * prev_state + self.alpha * np.tanh(np.dot(self.Win, x) + np.dot(self.W, prev_state))

    def fit(self, X, Y):
        # X: Input data (time_steps, input_size)
        # Y: Target data (time_steps, output_size)
        time_steps = X.shape[0]
        states = np.zeros((time_steps, self.reservoir_size))
        current_state = np.zeros(self.reservoir_size)

        for t in range(time_steps):
            current_state = self._update_reservoir(X[t], current_state)
            states[t] = current_state

        # Ridge regression to learn output weights
        self.Wout = Ridge(alpha=1e-6).fit(states, Y).coef_

    def predict(self, X):
        time_steps = X.shape[0]
        current_state = np.zeros(self.reservoir_size)
        predictions = np.zeros((time_steps, self.output_size))

        for t in range(time_steps):
            current_state = self._update_reservoir(X[t], current_state)
            predictions[t] = np.dot(self.Wout, current_state)

        return predictions

# # Example usage
# # Assuming input data is the past positions of the aircraft and output data is the future position to predict
# input_size = 3  # Example: [latitude, longitude, altitude]
# reservoir_size = 500
# output_size = 3
# time_steps = 100  # Number of time steps in the input sequence
#
# # Random example data for training
# X_train = np.random.rand(time_steps, input_size)
# Y_train = np.random.rand(time_steps, output_size)
#
# # Initialize and train ESN
# esn = ESN(input_size=input_size, reservoir_size=reservoir_size, output_size=output_size)
# esn.fit(X_train, Y_train)
#
# # Example prediction
# X_test = np.random.rand(time_steps, input_size)
# predictions = esn.predict(X_test)
#
# print(predictions)