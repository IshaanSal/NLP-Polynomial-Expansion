import sys
import numpy as np
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)

# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str):
    text_pairs = []
    factors, expansions = load_file("../research_assessment/nlp_assessment/train.txt")
    for i in range(len(factors)):
        text_pairs.append(factors[i])
        text_pairs.append(expansions[i])

    num_val_samples = int(0.20 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples:]

    vocab_size = 4000

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH + 1,
    )

    train_factored_texts = [pair[0] for pair in train_pairs]
    train_expansion_texts = [pair[1] for pair in train_pairs]
    source_vectorization.adapt(train_factored_texts)
    target_vectorization.adapt(train_expansion_texts)

    batch_size = 12

    def format_dataset(factored, expanded):
        factored = source_vectorization(factored)
        expanded = target_vectorization(expanded)
        return ({
                    "factored": factored,
                    "expanded": expanded[:, :-1], }, expanded[:, 1:])

    def make_dataset(pairs):
        factored_texts = zip(train_factored_texts)
        expanded_texts = zip(train_expansion_texts)
        factored_texts = list(factored_texts)
        expanded_texts = list(expanded_texts)
        dataset = tf.data.Dataset.from_tensor_slices((factored_texts, expanded_texts))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(format_dataset, num_parallel_calls=4)
        return dataset

    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)

    embed_dim = 50
    latent_dim = 400

    source = keras.Input(shape=(None,), dtype="int64", name="Factors")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
    encoded_source = layers.Bidirectional(layers.GRU(latent_dim), merge_mode="sum")(x)

    past_target = keras.Input(shape=(None,), dtype="int64", name="Expansions")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
    decoder_gru = layers.GRU(latent_dim, return_sequences=True)
    x = decoder_gru(x, initial_state=encoded_source)
    x = layers.Dropout(0.5)(x)
    target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
    seq2seq_rnn = keras.Model([source, past_target], target_next_step)

    seq2seq_rnn.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    seq2seq_rnn.fit(train_ds, epochs=7, validation_data=val_ds)
# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))

if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")