import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=0.1, epochs=20):
        self.w = np.zeros(input_dim)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(self.w, x) + self.b
        return self.activation(z)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                err = target - y_pred
                # perceptron güncellemesi
                self.w += self.lr * err * xi
                self.b += self.lr * err
        return self


def run_perceptron_demo():
    # 4 adet binary input
    X_logic = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    # 1) OR
    y_or = np.array([0, 1, 1, 1], dtype=float)
    or_p = Perceptron(input_dim=2).fit(X_logic, y_or)

    # 2) AND
    y_and = np.array([0, 0, 0, 1], dtype=float)
    and_p = Perceptron(input_dim=2).fit(X_logic, y_and)

    # 3) NAND
    y_nand = np.array([1, 1, 1, 0], dtype=float)
    nand_p = Perceptron(input_dim=2).fit(X_logic, y_nand)

    # XOR = (x1 NAND x2) AND (x1 OR x2)
    def xor_gate(x1, x2):
        out_nand = nand_p.predict(np.array([x1, x2]))
        out_or   = or_p.predict(np.array([x1, x2]))
        out_xor  = and_p.predict(np.array([out_nand, out_or]))
        return out_xor

    print("===== PART 3: PERCEPTRON LOGIC GATES =====")
    for a, b in X_logic:
        a = int(a); b = int(b)
        print(f"Inputs: {a} {b}")
        print(f"  OR:   {or_p.predict([a, b])}")
        print(f"  AND:  {and_p.predict([a, b])}")
        print(f"  NAND: {nand_p.predict([a, b])}")
        print(f"  XOR:  {xor_gate(a, b)}")
        print("-------------")
