"""
Exercise #1: Low-Rank Model Compression
In this exercise you will train a reasonably light-weight, dense, feed-forward neural network
on the standard MNIST dataset. After training, you will perform various degrees of low-rank
matrix approximation (SVD-based) on the weight matrices of this model, then perform
refinement training and finally report the test results of the compressed model(s).
"""

import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

# CONSTS 
BASELINE_EPOCHS = 100          # Always train baseline for 100 epochs
COMPRESSION_REFINEMENT_EPOCHS = 100  # Always refine compressed models for 100 epochs

# Force CPU only execution (ignore GPUs) to simplify environment.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

'''
UTILITIES 
'''
def set_seeds(seed=42):
    """Set RNG seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

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

def count_params(model):
    """Return total trainable parameter count."""
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))

"""
Step 1: Design/Train the Light-Weight Model

Train a light-weight, dense, feed-forward neural network (note: not a CNN) so that the first
hidden layers has 100 neurons, the next hidden layer has 50, and the final output layer has 10
neurons. All the layers should be fully-connected; layers should include conventional bias
neurons. Include a model summary with trainable parameter counts in your write-up.

Load and pre-process the MNIST dataset; report on the pre-processing procedure that you
apply in your assignment write-up.

Train your model on MNIST for 100 epochs, report the training/test loss and accuracy for
each epoch; include a 10x10 confusion matrix for the test data results on the fully trained
model.
"""
def step1(baseline_epochs=BASELINE_EPOCHS):
    """Wrapper for Step 1: returns (model, data, history, confusion_matrix)."""
    data = load_and_preprocess_mnist()
    model, hist, cm = step1_train_baseline(data, epochs=baseline_epochs)
    return model, data, hist, cm

def load_and_preprocess_mnist():
    """Load MNIST, scale to [0,1], flatten to 784 features."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') / 255.0).reshape(-1, 28*28)
    x_test  = (x_test.astype('float32')  / 255.0).reshape(-1, 28*28)
    return (x_train, y_train), (x_test, y_test)

def step1_train_baseline(data, epochs=BASELINE_EPOCHS):
    """Train baseline model for the specified number of epochs."""
    model = build_baseline_model()
    print("\nBaseline model summary")
    model.summary()
    print(f"Trainable params (baseline): {count_params(model)}")
    hist, cm = train_and_report(model, data, epochs, tag='Baseline(100-50-10)')
    return model, hist, cm


def build_baseline_model():
    """Create baseline dense network: 784 -> 100 -> 50 -> 10."""
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

"""
Step 2: Generate the Low-Rank Model

For each weight matrix in your model: 𝑊 (") , 𝑊 ($) , 𝑊 (%) , perform SVD (fine to use a SW
library function for this), so that 𝑊 (&) ≈ 𝑈 (&) 𝛴 (&) '𝑉 (&)) . To perform compression, your
SVD decomposition should represent a k-rank approximation to 𝑊 (&) ; in this way, if, say,
𝑊 (&) is of dimension 𝑚 × 𝑛, then 𝑈 (&) will be of dimension 𝑚 × 𝑘 (where 𝑘 < 𝑛), 𝛴 (&) is of
dimension 𝑘 × 𝑘 and '𝑉 (&) ) is of dimension 𝑘 × 𝑛.

For simplicity, as I have shown in lecture, express this decomposition as the product of two
matrices: 𝑊 (&) ≈ 𝑈′(&) '𝑉 (&) ) , where 𝑈′(&) = 𝑈 (&) 𝛴 (&) .

Define a new model – your compressed model – where each dense layer is from your 100-50-
10 model is replaced with two dense layers (representing the factors of your low-rank
approximation for each 𝑊 (&) ). 

I recommend using the Keras function “set_weights” to manually set the weights, after
defining your model, e.g.:
                            model.layers[ix].set_weights(A)

where above “𝑖𝑥” denotes the layer index and 𝐴 denotes the layer weight matrix. Note that
you can randomly initialize the layer biases in your new model or copy them from the
previously trained model – either method is fine, but please include details of your design
decisions in your assignment write-up. Include a model summary of your compressed model
with your assignment write-up.
"""
def svd_factorize(W, k):
    """Rank-k SVD factorization returning (U', V)."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    Uprime = U_k @ S_k  # (m,k)
    return Uprime.astype(np.float32), Vt_k.astype(np.float32)  # V is (k,n)

def compress_dense_to_two(dense_layer, k):
    """Factor Dense (m->n) into (m->k linear, no bias) + (k->n with activation & bias)."""
    W, b = dense_layer.get_weights()
    Uprime, V = svd_factorize(W, k)
    return Uprime, V, b.astype(np.float32)


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

def _compressed_model_name(ranks):
    """Build safe model name from rank list."""
    return 'compressed_' + '_'.join(str(r) for r in ranks)

"""
Step 3: Apply Refinement Training to the Low-Rank Model

Train your compressed model for 10 epochs. Report the training and test loss and accuracy for
each epoch. Include a 10x10 confusion matrix for the test data results on the fully trained model.

Note: Do not randomly initialize the weights of your compressed model prior to training.
Instead, use the weights learned from the low-rank approximation with the set_weights function,
as described in Step 2 above.
"""
def step3_refine_compressed(comp_model, data, epochs=COMPRESSION_REFINEMENT_EPOCHS, tag='Compressed'):
    """Fine-tune compressed model for given epochs."""
    return train_and_report(comp_model, data, epochs, tag)

"""
Step 4: Apply Different Degrees of Compression

Execute Steps 2 and 3 above at 2x, 4x, and 8x compression, respectively. 
For example, with 2x compression, if W is of dimension m * n (uncompressed), 
apply SVD-based compression with k = int(n / 2). For 4x compression, set k = int(n / 4), 
and for 8x compression, k = int(n / 8). 

For each compression level, include a model summary and trainable parameter count.
"""

def step4(baseline_model, data, baseline_params=None, factors=(2,4,8), comp_epochs=COMPRESSION_REFINEMENT_EPOCHS):
    """Apply Steps 2 & 3 for each factor, collecting metrics/results."""
    if baseline_params is None:
        baseline_params = count_params(baseline_model)
    return step4_run_all_compressions(baseline_model, data, baseline_params, factors=factors, comp_epochs=comp_epochs)


def step4_run_all_compressions(baseline_model, data, baseline_params, factors=(2,4,8), comp_epochs=COMPRESSION_REFINEMENT_EPOCHS):
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

def compute_ranks_for_factor(factor):
    """Compute ranks [k1,k2,k3] for a compression factor."""
    return [k_for_compression(100, factor),
            k_for_compression(50, factor),
            k_for_compression(10, factor)]

def k_for_compression(n, factor):
    """Compute rank k = int(n/factor) with lower bound 1."""
    return max(1, int(n / factor))

#Pipeline
def main():
    """Run all steps sequentially and return collected results."""
    set_seeds(42)
    print("Step 1") 
    baseline_model, data, base_hist, base_cm = step1(BASELINE_EPOCHS)
    baseline_params = count_params(baseline_model)
    compression_results = step4(baseline_model, data, baseline_params, factors=(2,4,8), comp_epochs=COMPRESSION_REFINEMENT_EPOCHS)
    return {
        'baseline_history': base_hist.history,
        'baseline_confusion_matrix': base_cm,
        'compression_results': compression_results
    }

if __name__ == '__main__':
    main()