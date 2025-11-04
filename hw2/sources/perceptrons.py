import numpy as np
from .custom_metrics import Metrics


class Perceptron:
    """
    Simple single-layer perceptron for binary outputs.
    """

    def __init__(self, input_dim, lr, epochs):
        #initializing weights and bias to 0
        #weights could be initialized randomly (eg: xaiver method) but for simplicity we initialize them to 0
        self.w = np.zeros(input_dim, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs


    def step_activation(self, z):
        return 1 if z >= 0 else 0 #binary step function default activation function of perceptrons

    def predict(self, x):
        x = np.array(x, dtype=float)
        z = np.dot(self.w, x) + self.b #for percepeptrons outputs are linear combination of weights and inputs + biases
        return self.step_activation(z)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        for epoch in range(self.epochs):
            print("Epoch: ", epoch, "of training")
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                err = target - y_pred
                if err != 0:
                    # perceptron update
                    self.w += self.lr * err * xi
                    self.b += self.lr * err
        return self


def train_gate(X_logic,y_gate,lr,epochs):
    """
    Helper that mirrors the 'fit' style in KNN/CNN files but for logic gates.
    """
    p = Perceptron(input_dim=2, lr=lr, epochs=epochs)
    p.fit(X_logic, y_gate)
    return p

def xor_gate(x1, x2,nand_gate_model,and_gate_model,or_gate_model):
        inp = np.array([x1, x2], dtype=float)
        out_nand = nand_gate_model.predict(inp)  # 0/1
        out_or   = or_gate_model.predict(inp)    # 0/1
        # final AND takes these 2 as inputs
        return and_gate_model.predict(np.array([out_nand, out_or], dtype=float))
def part3_gate_models():
    """
    Part 3 (extra credit):
      1. Train OR, AND, NAND perceptrons on truth tables
      2. Define XOR = AND( NAND(x1,x2), OR(x1,x2) )
      3. Evaluate XOR with our own custom_metrics (no sklearn)
    """
    # 4 input combinations
    X_logic = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    # target tables
    y_or   = np.array([0, 1, 1, 1], dtype=float)
    y_and  = np.array([0, 0, 0, 1], dtype=float)
    y_nand = np.array([1, 1, 1, 0], dtype=float)

    # 1) train gates
    print("==============Training of OR Gate==============")
    or_gate_model = train_gate(X_logic, y_or,   lr=0.01, epochs=30)
    print("==============Training of AND Gate==============")
    and_gate_model = train_gate(X_logic, y_and,  lr=0.01, epochs=30)
    print("==============Training of NAND Gate==============")
    nand_gate_model = train_gate(X_logic, y_nand, lr=0.01, epochs=30)

    # 2) build XOR using trained gates
    

    # 3) test + metrics (like KNN, but with our own Metrics class)
    print("\n===== PART 3: PERCEPTRON LOGIC GATES =====\n")
    y_true_xor = []
    y_pred_xor = []

    for a, b in X_logic:
        a = int(a)
        b = int(b)
        inp = np.array([a, b], dtype=float)

        pred_or   = or_gate_model.predict(inp)
        pred_and  = and_gate_model.predict(inp)
        pred_nand = nand_gate_model.predict(inp)
        pred_xor  = xor_gate(a, b,nand_gate_model,and_gate_model,or_gate_model)

    
        true_xor = (a ^ b)  # ground truth for XOR calculated for covariance matrix

        y_true_xor.append(true_xor)
        y_pred_xor.append(pred_xor)

        print(f"Inputs: {a} {b}")
        print(f"  OR   = {pred_or}")
        print(f"  AND  = {pred_and}")
        print(f"  NAND = {pred_nand}")
        print(f"  XOR  = {pred_xor}")
        print("--------------")

    y_true_xor = np.array(y_true_xor, dtype=int)
    y_pred_xor = np.array(y_pred_xor, dtype=int)

    # our Metrics assumes labels start from 1 (it does true-1, pred-1) so shift {0,1} -> {1,2}
    y_true_shifted = y_true_xor + 1
    y_pred_shifted = y_pred_xor + 1

    cm = Metrics.confusion_matrix(y_true_shifted, y_pred_shifted, num_classes=2)
    metric_results, accuracy = Metrics.calculate_metrics(cm, num_classes=2)

    print("\n--- XOR confusion matrix (using custom_metrics) ---")
    print(cm)
    print(f"Accuracy: {accuracy:.2f}")
    print("Per-class metrics:")
    for cls_name, vals in metric_results.items():
        # cls_name will be 'class_0' and 'class_1' because we passed num_classes=2
        print(f"  {cls_name}: "
              f"precision={vals['precision']:.2f}, "
              f"recall={vals['recall']:.2f}, "
              f"f1={vals['f1_score']:.2f}")
