import os, sys, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

# ================== Utility & Environment ==================
# Force CPU only execution (ignore GPUs) to simplify environment.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

## Support Utility
# Set deterministic seeds for reproducibility.
def set_seeds(seed=42):
    """Set RNG seeds."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ================== Step 1: Data & Baseline Model ==================
## Step 1: Data Loading & Preprocessing
# Load MNIST, scale to [0,1], flatten to 784 features.
def load_and_preprocess_mnist():
    """Load & preprocess MNIST."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') / 255.0).reshape(-1, 28*28)
    x_test  = (x_test.astype('float32')  / 255.0).reshape(-1, 28*28)
    return (x_train, y_train), (x_test, y_test)

## Step 1: Baseline Model Definition (784 -> 100 -> 50 -> 10)
def build_baseline_model():
    """Create baseline dense network."""
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

## Steps 1 & 3: Training / Reporting Helper
def train_and_report(model, data, epochs, tag):
    """Train model; emit per-epoch metrics & confusion matrix."""
    (x_train, y_train), (x_test, y_test) = data
    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                     epochs=epochs, batch_size=128, verbose=2)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== {tag}: epoch-by-epoch metrics ===")
    for i,(trL,trA,vlL,vlA) in enumerate(zip(hist.history['loss'],
                                            hist.history['accuracy'],
                                            hist.history['val_loss'],
                                            hist.history['val_accuracy']),1):
        print(f"Epoch {i:03d}: loss={trL:.4f} acc={trA:.4f} | val_loss={vlL:.4f} val_acc={vlA:.4f}")
    print(f"\n{tag} — Confusion Matrix (10x10):\n{cm}\n")
    return hist, cm

## Support: Parameter Counting
def count_params(model):
    """Return total trainable parameter count."""
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))

## Step 1: Design/Train the Light-Weight Model
def step1_train_baseline(data, epochs=100):
    """Train baseline and return (model, history, confusion matrix)."""
    model = build_baseline_model()
    print("\nBaseline model summary ===")
    model.summary()
    print(f"Trainable params (baseline): {count_params(model)}")
    hist, cm = train_and_report(model, data, epochs, tag='Baseline(100-50-10)')
    return model, hist, cm

## Step 1 Wrapper
def step1(baseline_epochs=100):
        """Step 1: Design / Train the Light-Weight Model.

        Actions:
            1. Load & preprocess MNIST (scale to [0,1], flatten 28x28 -> 784).
            2. Build 784->100->50->10 dense network (bias on every Dense, ReLU hidden, softmax output).
            3. Train for `baseline_epochs` (default 100) capturing per-epoch train & validation metrics.
            4. Produce confusion matrix for test set.

        Returns:
            (baseline_model, data_tuple, history, confusion_matrix)
        """
        data = load_and_preprocess_mnist()
        model, hist, cm = step1_train_baseline(data, epochs=baseline_epochs)
        return model, data, hist, cm

# ================== Step 2: Low-Rank Decomposition (SVD) ==================
## Step 2: Low-Rank Decomposition (SVD)
def svd_factorize(W, k):
    """Rank-k SVD factorization returning (U', V)."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    Uprime = U_k @ S_k  # (m,k)
    return Uprime.astype(np.float32), Vt_k.astype(np.float32)  # V is (k,n)

## Step 2: Dense Layer Factorization Helper
def compress_dense_to_two(dense_layer, k):
    """Factor Dense (m->n) into (m->k linear no-bias) + (k->n with activation & bias)."""
    W, b = dense_layer.get_weights()
    Uprime, V = svd_factorize(W, k)
    return Uprime, V, b.astype(np.float32)

## Support: Naming Helper
def _compressed_model_name(ranks):
    """Build safe model name from rank list."""
    return 'compressed_' + '_'.join(str(r) for r in ranks)

## Step 2: Construct Compressed Model From Baseline
def build_compressed_model(baseline_model, ranks):
    """Build compressed model using ranks [k1,k2,k3]."""
    assert len(ranks) == 3
    d_layers = [l for l in baseline_model.layers if isinstance(l, layers.Dense)]
    (d1, d2, d3) = d_layers
    W1a, W1b, b1 = compress_dense_to_two(d1, ranks[0])
    W2a, W2b, b2 = compress_dense_to_two(d2, ranks[1])
    W3a, W3b, b3 = compress_dense_to_two(d3, ranks[2])
    inp = keras.Input(shape=(784,))
    z = layers.Dense(W1a.shape[1], activation='linear', use_bias=False, name='proj1')(inp)
    z = layers.Dense(W1b.shape[1], activation='relu', use_bias=True, name='rec1')(z)
    z = layers.Dense(W2a.shape[1], activation='linear', use_bias=False, name='proj2')(z)
    z = layers.Dense(W2b.shape[1], activation='relu', use_bias=True, name='rec2')(z)
    z = layers.Dense(W3a.shape[1], activation='linear', use_bias=False, name='proj3')(z)
    out = layers.Dense(W3b.shape[1], activation='softmax', use_bias=True, name='rec3')(z)
    comp = keras.Model(inp, out, name=_compressed_model_name(ranks))
    comp.compile(optimizer=keras.optimizers.Adam(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    # set weights
    name_to_layer = {l.name: l for l in comp.layers if isinstance(l, layers.Dense)}
    name_to_layer['proj1'].set_weights([W1a])
    name_to_layer['rec1'].set_weights([W1b, b1])
    name_to_layer['proj2'].set_weights([W2a])
    name_to_layer['rec2'].set_weights([W2b, b2])
    name_to_layer['proj3'].set_weights([W3a])
    name_to_layer['rec3'].set_weights([W3b, b3])
    return comp

## Step 2 Wrapper
def step2(baseline_model, factor=2):
    """Step 2: Generate a Low-Rank (SVD) Compressed Model for one compression factor.

    Computes per-layer ranks k = int(n/factor) (with lower bound 1) for n in (100,50,10), builds
    and returns the compressed model plus the rank list.
    """
    ranks = [max(1, int(100/factor)), max(1, int(50/factor)), max(1, int(10/factor))]
    comp = build_compressed_model(baseline_model, ranks)
    return comp, ranks

# ================== Step 3: Refinement Training ==================
## Step 3: Refinement Training
def step3_refine_compressed(comp_model, data, epochs=10, tag='Compressed'):
    """Fine-tune compressed model for given epochs."""
    return train_and_report(comp_model, data, epochs, tag)

## Step 3 Wrapper
def step3(compressed_model, data, refine_epochs=10, tag='Compressed'):
    """Step 3: Refinement (fine-tuning) of previously constructed compressed model."""
    return step3_refine_compressed(compressed_model, data, epochs=refine_epochs, tag=tag)

# ================== Step 4: Multiple Compression Levels ==================
## Step 4: Rank Calculation Helper
def k_for_compression(n, factor):
    """Compute rank k = int(n/factor) with lower bound 1."""
    return max(1, int(n / factor))

## Step 4: Ranks for Entire Network
def compute_ranks_for_factor(factor):
    """Compute ranks [k1,k2,k3] for a compression factor."""
    return [k_for_compression(100, factor),
            k_for_compression(50, factor),
            k_for_compression(10, factor)]

## Step 4: Run All Compression Factors (2x, 4x, 8x)
def step4_run_all_compressions(baseline_model, data, baseline_params, factors=(2,4,8), comp_epochs=10):
    """For each factor perform low-rank construction + refinement; collect results."""
    results = {}
    print("\n=== Compression Plan (factor -> ranks) ===")
    for f in factors:
        ranks = compute_ranks_for_factor(f)
        print(f"{f}x -> {ranks}")
    for f in factors:
        ranks = compute_ranks_for_factor(f)
        tag = f"Compressed-{f}x"
        print(f"\n=== Building compressed model {tag} (ranks {ranks}) ===")
        comp = build_compressed_model(baseline_model, ranks)
        comp.summary()
        comp_params = count_params(comp)
        ratio = comp_params / baseline_params
        print(f"Trainable params ({tag}): {comp_params} (ratio {ratio:.3f} vs baseline {baseline_params})")
        hist, cm = step3_refine_compressed(comp, data, epochs=comp_epochs, tag=tag)
        results[tag] = {
            'ranks': ranks,
            'params': comp_params,
            'ratio': ratio,
            'history': hist.history,
            'confusion_matrix': cm
        }
    return results

## Step 4 Wrapper
def step4(baseline_model, data, baseline_params=None, factors=(2,4,8), comp_epochs=10):
    """Step 4: Apply Different Degrees of Compression (2x, 4x, 8x).

    Executes Steps 2 & 3 for each factor, reporting ranks, parameter counts, metrics, confusion matrix.
    """
    if baseline_params is None:
        baseline_params = count_params(baseline_model)
    return step4_run_all_compressions(baseline_model, data, baseline_params, factors=factors, comp_epochs=comp_epochs)

# ================== Orchestrator ==================
## Orchestrator: Executes Steps 1-4 Sequentially
def main():
    """Run baseline training then all compression/refinement runs; return collected results."""
    set_seeds(42)
    baseline_model, data, base_hist, base_cm = step1(baseline_epochs=100)
    baseline_params = count_params(baseline_model)  # after step1
    compression_results = step4(baseline_model, data, baseline_params, factors=(2,4,8), comp_epochs=100)
    return {
        'baseline_history': base_hist.history,
        'baseline_confusion_matrix': base_cm,
        'compression_results': compression_results
    }

if __name__ == '__main__':
    main()
