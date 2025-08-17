import os, math, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
np.random.seed(42); tf.random.set_seed(42)

# ---------- Data ----------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Preprocess: scale to [0,1], flatten to 784
x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28*28)
x_test  = (x_test.astype("float32")  / 255.0).reshape(-1, 28*28)

# ---------- Baseline model: 784 -> 100 -> 50 -> 10 ----------
def build_baseline():
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(100, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_and_eval(model, epochs, tag):
    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                     epochs=epochs, batch_size=128, verbose=2)
    # predictions & confusion matrix
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    # print loss/acc per epoch (you can copy into your report)
    print(f"\n=== {tag}: epoch-by-epoch metrics ===")
    for i,(trL,trA,vlL,vlA) in enumerate(zip(hist.history['loss'],
                                            hist.history['accuracy'],
                                            hist.history['val_loss'],
                                            hist.history['val_accuracy']), 1):
        print(f"Epoch {i:03d}: loss={trL:.4f} acc={trA:.4f} | "
              f"val_loss={vlL:.4f} val_acc={vlA:.4f}")
    print(f"\n{tag} — Confusion Matrix (10x10):\n{cm}\n")
    return hist, cm

def count_params(model):
    return np.sum([np.prod(v.shape) for v in model.trainable_variables])

# Train baseline for 100 epochs
baseline = build_baseline()
print("\n=== Baseline model summary ===")
baseline.summary()
print(f"Trainable params (baseline): {count_params(baseline)}")
hist_base, cm_base = train_and_eval(baseline, epochs=100, tag="Baseline(100-50-10)")

# ---------- SVD utilities ----------
def svd_factorize(W, k):
    # W ≈ U_k @ S_k @ Vt_k; pack as (U' = U_k @ S_k) and V = Vt_k
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    Uprime = U_k @ S_k      # shape: [m, k]
    V = Vt_k                 # shape: [k, n] (we'll use V.T when setting a Dense)
    return Uprime.astype(np.float32), V.astype(np.float32)  # V is [k, n]

def compress_dense_to_two(dense_layer, k):
    """Return (W1, b1, W2, b2) that replaces Dense(W,b) with Dense(k)->Dense(n)"""
    W, b = dense_layer.get_weights()
    m, n = W.shape  # input dim m, output dim n (Keras stores [in, out])
    # BUT Keras Dense kernel is shape (input_dim, units) = (m, n) indeed.
    # We want W^T factorization for mapping: y = x @ W + b
    # If W = U' @ V  where U' in R^{m x k}, V in R^{k x n}
    # Implement as: x -> Dense(k) with kernel U' (no bias) -> Dense(n) with kernel V (with original bias)
    Uprime, V = svd_factorize(W, k)
    W1 = Uprime  # (m, k): used by first Dense to map x(m) -> k
    W2 = V       # (k, n): second Dense maps k -> n
    # Bias handling: assign original bias to the second Dense
    b1 = np.zeros((k,), dtype=np.float32)
    b2 = b.astype(np.float32)
    return W1, b1, W2, b2

def build_compressed_from(baseline_model, ranks):
    """
    Replace each Dense with two Dense layers using given ranks (list of k per original dense).
    ranks should match the three original Dense layers: [k1, k2, k3]
    """
    assert len(ranks) == 3
    # Extract original layers
    d1, d2, d3 = [l for l in baseline_model.layers if isinstance(l, layers.Dense)]
    # Factorize
    W1a, b1a, W1b, b1b = compress_dense_to_two(d1, ranks[0])
    W2a, b2a, W2b, b2b = compress_dense_to_two(d2, ranks[1])
    W3a, b3a, W3b, b3b = compress_dense_to_two(d3, ranks[2])

    # Build compressed model
    inputs = keras.Input(shape=(784,))
    z = layers.Dense(W1a.shape[1], activation="linear", use_bias=True)(inputs)    # k1
    z = layers.ReLU()(z)
    z = layers.Dense(W1b.shape[1], activation="linear", use_bias=True)(z)         # 100
    z = layers.ReLU()(z)
    z = layers.Dense(W2a.shape[1], activation="linear", use_bias=True)(z)         # k2
    z = layers.ReLU()(z)
    z = layers.Dense(W2b.shape[1], activation="linear", use_bias=True)(z)         # 50
    z = layers.ReLU()(z)
    z = layers.Dense(W3a.shape[1], activation="linear", use_bias=True)(z)         # k3
    z = layers.ReLU()(z)
    outputs = layers.Dense(W3b.shape[1], activation="softmax", use_bias=True)(z)  # 10
    comp = keras.Model(inputs, outputs)
    comp.compile(optimizer=keras.optimizers.Adam(),
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])

    # Set weights in order (Dense layers appear in same sequence as built)
    dense_layers = [l for l in comp.layers if isinstance(l, layers.Dense)]
    # Assign (kernel,bias) per layer
    sets = [
        (W1a, b1a), (W1b, b1b),
        (W2a, b2a), (W2b, b2b),
        (W3a, b3a), (W3b, b3b)
    ]
    for L, (W, b) in zip(dense_layers, sets):
        L.set_weights([W, b])
    return comp

# ---------- Choose ranks for 2×, 4×, 8× ----------
# Original layer sizes: 784->100, 100->50, 50->10  (kernels: 784x100, 100x50, 50x10)
def k_for_compression(n, factor):
    # from spec: k = int(n / factor)  (use output dimension n of each Dense)
    return max(1, int(n / factor))

compressions = {
    "2x": [k_for_compression(100,2), k_for_compression(50,2), k_for_compression(10,2)],
    "4x": [k_for_compression(100,4), k_for_compression(50,4), k_for_compression(10,4)],
    "8x": [k_for_compression(100,8), k_for_compression(50,8), k_for_compression(10,8)],
}

for tag, ranks in compressions.items():
    print(f"\n=== Building compressed model {tag} with ranks {ranks} ===")
    comp = build_compressed_from(baseline, ranks)
    print(f"\n{tag} compressed model summary:")
    comp.summary()
    print(f"Trainable params ({tag}): {count_params(comp)}")
    # Fine-tune 10 epochs
    _, cm = train_and_eval(comp, epochs=10, tag=f"Compressed-{tag}")
