import argparse
import numpy as np

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def bce_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def accuracy(y, p):
    return float(np.mean((p >= 0.5) == y))

class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        lim1 = np.sqrt(6 / (in_dim + hidden_dim))
        lim2 = np.sqrt(6 / (hidden_dim + out_dim))
        self.W1 = rng.uniform(-lim1, lim1, (in_dim, hidden_dim))
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = rng.uniform(-lim2, lim2, (hidden_dim, out_dim))
        self.b2 = np.zeros((out_dim,))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        p = sigmoid(z2)
        return a1, p

    def backward(self, X, a1, p, y):
        N = X.shape[0]
        dz2 = (p - y) / N
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - a1**2)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return dW1, db1, dW2, db2

    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

def make_xor_dataset(noise_std=0.0, augment=1, seed=0):
    X0 = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=np.float32)
    y0 = np.array([[0.],[1.],[1.],[0.]], dtype=np.float32)
    if augment <= 1:
        return X0, y0
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for x, y in zip(X0, y0):
        xs = np.tile(x, (augment, 1)).astype(np.float32)
        if noise_std > 0:
            xs += rng.normal(0.0, noise_std, xs.shape).astype(np.float32)
        ys.append(np.tile(y, (augment, 1)).astype(np.float32))
        Xs.append(xs)
    return np.vstack(Xs), np.vstack(ys)

def train(model, X, y, lr, epochs, log_every=500):
    for ep in range(1, epochs + 1):
        a1, p = model.forward(X)
        loss = bce_loss(y, p)
        grads = model.backward(X, a1, p, y)
        model.step(grads, lr)
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            print(f"Epoch {ep}/{epochs}  loss={loss:.6f}  acc={accuracy(y, p)*100:.2f}%")
    return model

def ascii_boundary(model, xmin=-0.5, xmax=1.5, ymin=-0.5, ymax=1.5, steps=21):
    xs = np.linspace(xmin, xmax, steps)
    ys = np.linspace(ymin, ymax, steps)[::-1]
    print("\nASCII decision boundary (#=1, .=0)")
    print("y ↑")
    for y in ys:
        row = []
        for x in xs:
            _, p = model.forward(np.array([[x, y]], dtype=np.float32))
            row.append('#' if p[0,0] >= 0.5 else '.')
        print(''.join(row))
    print("   " + "-"*steps + "→ x")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Tiny NumPy MLP for XOR with manual backprop.")
    ap.add_argument("--hidden", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--augment", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--no-interactive", action="store_true")
    args = ap.parse_args(argv)

    X, y = make_xor_dataset(args.noise, args.augment, args.seed)
    print(f"Dataset: X={X.shape}, y={y.shape}")

    model = MLP(2, args.hidden, 1, seed=args.seed)
    train(model, X, y, lr=args.lr, epochs=args.epochs, log_every=max(1, args.epochs//10))

    if args.grid:
        ascii_boundary(model)

    if not args.no_interactive:
        print("\nEnter two numbers in [0,1], e.g., '1 0'. Type 'q' to quit.")
        while True:
            try:
                s = input("> ").strip()
            except EOFError:
                break
            if s.lower() in {"q", "quit", "exit"}:
                break
            if not s:
                continue
            try:
                a, b = map(float, s.split())
            except ValueError:
                print("Example: 0 1")
                continue
            _, p = model.forward(np.array([[a, b]], dtype=np.float32))
            prob = float(p[0,0])
            print(f"P(class=1)={prob:.4f} -> class {int(prob>=0.5)}")

if __name__ == "__main__":
    main() 