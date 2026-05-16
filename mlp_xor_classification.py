import numpy as np

# -----------------------------
# Hyper-parameters
# -----------------------------
np.random.seed(42)  # reproducibility
eta = 0.5  # learning-rate
epochs = 2

# -----------------------------
# Dataset (XOR)
# -----------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)  # shape (4, 2)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)  # shape (4, 1)

# -----------------------------
# MLP architecture (2-2-1)
# -----------------------------
n_inputs = 2
n_hidden = 2
n_outputs = 1

# Weight matrices
W1 = np.random.uniform(-1, 1, size=(n_inputs, n_hidden))  # shape (2,2)
b1 = np.zeros((1, n_hidden))  # shape (1,2)
W2 = np.random.uniform(-1, 1, size=(n_hidden, n_outputs))  # shape (2,1)
b2 = np.zeros((1, n_outputs))  # shape (1,1)

# -----------------------------
# Helper functions
# -----------------------------
def sigmoid(z):  # activation
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):  # derivative w.r.t. activation value
    return a * (1 - a)

# -----------------------------
# Training loop (online SGD)
# -----------------------------
for epoch in range(1, epochs + 1):
    for x_vec, target in zip(X, y):
        x_vec = x_vec.reshape(1, -1)  # shape (1,2)
        target = target.reshape(1, -1)  # shape (1,1)

        # ---- Forward pass ----
        z1 = x_vec @ W1 + b1  # (1,2)
        a1 = sigmoid(z1)      # hidden layer output
        z2 = a1 @ W2 + b2     # (1,1)
        a2 = sigmoid(z2)      # network prediction

        # ---- Backward pass ----
        error_out = a2 - target                 # (1,1)
        delta_out = error_out * sigmoid_deriv(a2)  # (1,1)
        error_hidden = delta_out @ W2.T         # (1,2)
        delta_hidden = error_hidden * sigmoid_deriv(a1)  # (1,2)

        # ---- Weight & bias updates ----
        W2 -= eta * a1.T @ delta_out  # (2,1)
        b2 -= eta * delta_out         # (1,1)
        W1 -= eta * x_vec.T @ delta_hidden  # (2,2)
        b1 -= eta * delta_hidden       # (1,2)

    # ---- Epoch summary ----
    print(f"\nEpoch {epoch} complete")
    print("W1 =\n", W1)
    print("b1 =", b1)
    print("W2 =\n", W2)
    print("b2 =", b2)

# -----------------------------
# Quick evaluation on training set
# -----------------------------
def predict(x):
    a1 = sigmoid(x @ W1 + b1) S
    a2 = sigmoid(a1 @ W2 + b2)
    return (a2 > 0.5).astype(int)

print("\nPredictions after 2 epochs:")
for xv, target in zip(X, y):
    print(f"{xv} → {predict(xv)} (target {int(target[0])})")

