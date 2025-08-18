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
BASELINE_EPOCHS = 100 # Train your baseline model for 100 epochs; 
REFINEMENT_EPOCHS = 10  # Train your compressed model for 10 epochs; 

# Doing this since my gpu doesn't work for some reason, 5080 is not working
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

'''
UTILITIES 
'''

def set_seeds(seed=420):
    """Random seed using funny number"""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_print_progress(model, data, epochs, tag):
    """Train model; emit per-epoch metrics & confusion matrix."""
    (x_train, y_train), (x_test, y_test) = data
    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                     epochs=epochs, batch_size=128, verbose=2)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{tag}: Epoch Training")

    # Fancy python formatting
    for i,(trL,trA,vlL,vlA) in enumerate(zip(hist.history['loss'],
                                            hist.history['accuracy'],
                                            hist.history['val_loss'],
                                            hist.history['val_accuracy']),1):

        print(f"Epoch # {i}: loss={trL:.4f} acc={trA:.4f} | val_loss={vlL:.4f} val_acc={vlA:.4f}")
    print(f"\n{tag} — Confusion Matrix (10x10):\n{cm}\n")
    return hist, cm

def count_params(model):
    """Return total trainable parameter count."""
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))

#################################################################

def step1(baseline_epochs=BASELINE_EPOCHS):
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
    data = load_and_preprocess_mnist()
    model, hist, cm = step1_train_baseline(data, epochs=baseline_epochs)
    return model, data, hist, cm

def load_and_preprocess_mnist():
    """
    Load and pre-process the MNIST dataset; report on the pre-processing procedure that you
    apply in your assignment write-up.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') / 255.0).reshape(-1, 28*28)
    x_test  = (x_test.astype('float32')  / 255.0).reshape(-1, 28*28)
    return (x_train, y_train), (x_test, y_test)

def step1_train_baseline(data, epochs=BASELINE_EPOCHS):
    model = build_baseline_model()
    print("**************************************************************************")
    print("\nInclude a model summary with trainable parameter counts in your write-up.")
    model.summary()
    print("**************************************************************************")
    print(f"Trainable params (baseline): {count_params(model)}")
    hist, cm = train_print_progress(model, data, epochs, tag='Baseline(100-50-10)')
    return model, hist, cm


def build_baseline_model():
    """
    Train a light-weight, dense, feed-forward neural network (note: not a CNN) so that the first
    hidden layers has 100 neurons, the next hidden layer has 50, and the final output layer has 10
    neurons. All the layers should be fully-connected; layers should include conventional bias
    neurons. Include a model summary with trainable parameter counts in your write-up.
    """
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(100, activation='relu'), #hidden layer 100 neurons
        layers.Dense(50, activation='relu'), #hidden layer 50 neurons
        layers.Dense(10, activation='softmax') #output layer 10 neurons
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#################################################################

def step_2_build_compressed_model(baseline_model, ranks):
    """
    Step 2: Generate the Low-Rank Model

    For each weight matrix in your model: 𝑊 (1) , 𝑊 (2) , 𝑊 (3) , perform SVD (fine to use a SW
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
    assert len(ranks) == 3

    # Extract Dense layers from the baseline model
    dense_layers = [layer for layer in baseline_model.layers if isinstance(layer, layers.Dense)]
    d1, d2, d3 = dense_layers

    # Compress each dense layer using SVD-based factorization
    # For each weight matrix in your model: 𝑊 (1) , 𝑊 (2) , 𝑊 (3) , perform SVD
    W1a, W1b, b1 = compress_dense_to_two(d1, ranks[0])
    W2a, W2b, b2 = compress_dense_to_two(d2, ranks[1])
    W3a, W3b, b3 = compress_dense_to_two(d3, ranks[2])

    # Build the compressed model using pairs of Dense layers for each original layer
    inp = keras.Input(shape=(784,))
    x = layers.Dense(W1a.shape[1], activation='linear', use_bias=False, name='proj1')(inp)
    x = layers.Dense(W1b.shape[1], activation='relu', use_bias=True, name='rec1')(x)
    x = layers.Dense(W2a.shape[1], activation='linear', use_bias=False, name='proj2')(x)
    x = layers.Dense(W2b.shape[1], activation='relu', use_bias=True, name='rec2')(x)
    x = layers.Dense(W3a.shape[1], activation='linear', use_bias=False, name='proj3')(x)
    out = layers.Dense(W3b.shape[1], activation='softmax', use_bias=True, name='rec3')(x)

    comp_model = keras.Model(inp, out, name=_compressed_model_name(ranks))
    comp_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Set the weights for each layer in the compressed model
    # Using the recommended set_weights function.
    layer_map = {layer.name: layer for layer in comp_model.layers if isinstance(layer, layers.Dense)}
    layer_map['proj1'].set_weights([W1a])
    layer_map['rec1'].set_weights([W1b, b1])
    layer_map['proj2'].set_weights([W2a])
    layer_map['rec2'].set_weights([W2b, b2])
    layer_map['proj3'].set_weights([W3a])
    layer_map['rec3'].set_weights([W3b, b3])

    return comp_model

def perform_svd_decomposition(W, k):
    """
    For each weight matrix in your model: 𝑊 (") , 𝑊 ($) , 𝑊 (%) , perform SVD.
    matrices: 𝑊 (&) ≈ 𝑈′(&) '𝑉 (&) ) , where 𝑈′(&) = 𝑈 (&) 𝛴 (&) .
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    Uprime = U_k @ S_k  # (m,k)
    return Uprime.astype(np.float32), Vt_k.astype(np.float32)  # V is (k,n)

def compress_dense_to_two(dense_layer, k):
    """Compressing the dense layer using SVD."""
    W, b = dense_layer.get_weights()
    Uprime, V = perform_svd_decomposition(W, k)
    return Uprime, V, b.astype(np.float32)

def _compressed_model_name(ranks):
    """Build safe model name from rank list."""
    return 'compressed_' + '_'.join(str(r) for r in ranks)


#################################################################

def step4(baseline_model, data, baseline_params=None, factors=(2,4,8), comp_epochs=REFINEMENT_EPOCHS):
    """
    Step 4: Apply Different Degrees of Compression

    Execute Steps 2 and 3 above at 2x, 4x, and 8x compression, respectively. 
    For example, with 2x compression, if W is of dimension m * n (uncompressed), 
    apply SVD-based compression with k = int(n / 2). For 4x compression, set k = int(n / 4), 
    and for 8x compression, k = int(n / 8). 

    For each compression level, include a model summary and trainable parameter count.
    """
    print("**************************************************************************")
    if baseline_params is None:
        baseline_params = count_params(baseline_model)
    results = {}
    for f in factors:
        ranks = compute_ranks_for_factor(f)
        print(f"{f}x {ranks}")
    for f in factors:
        ranks = compute_ranks_for_factor(f)
        tag = f"Compressed-{f}x"
        comp = step_2_build_compressed_model(baseline_model, ranks)
        print("**************************************************************************")
        comp.summary()
        print("**************************************************************************")
        comp_params = count_params(comp)
        ratio = comp_params / baseline_params
        print(f"params ({tag}): {comp_params}, ratio {ratio} vs baseline {baseline_params}")
        # PART 3: refinement training
        hist, cm = train_print_progress(comp, data, epochs=comp_epochs, tag=tag)
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
    set_seeds()
    print("Step 1") 
    baseline_model, data, base_hist, base_cm = step1(BASELINE_EPOCHS)
    baseline_params = count_params(baseline_model)
    compression_results = step4(baseline_model, data, baseline_params, factors=(2,4,8), comp_epochs=REFINEMENT_EPOCHS)
    return {
        'baseline_history': base_hist.history,
        'baseline_confusion_matrix': base_cm,
        'compression_results': compression_results
    }

if __name__ == '__main__':
    main()