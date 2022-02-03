import argparse
from random import choice
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer, Dense, LSTM, Flatten, TimeDistributed
from tensorflow.keras.layers import RepeatVector, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam


max_query_length = 7
max_answer_length = 4
unique_characters = "0123456789+- "


class Encoder(Layer):
    """ The section of the model that encodes the batch of onehot encoded matrices
        to a matrix of latent vectors.
    """

    def __init__(self, latent_dim=128):
        """ latent_dim: dimension of the space where the input text is encoded to
        """
        super().__init__()
        
        # instantiate layers
        self.lstm_1 = LSTM(latent_dim, return_sequences=True)
        self.lstm_2 = LSTM(latent_dim, return_sequences=True)
        self.lstm_3 = LSTM(latent_dim)

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        return self.lstm_3(x)


class Decoder(Layer):
    """ The final part of the model responsible for upscaling each latent vector
        to a sequence of 4 28x28 images.
    """

    def __init__(self):
        super().__init__()

        # instantiate layers
        self.repeat = RepeatVector(4)
        self.lstm_1 = LSTM(256, return_sequences=True)
        self.lstm_2 = LSTM(256, return_sequences=True)
        self.time_distr_dense = TimeDistributed(Dense(1024, activation="relu"))
        self.flatten = Flatten()
        self.reshape = Reshape((64, 8, 8))
        self.conv2d_transp_1 = Conv2DTranspose(8, 5, strides=1, data_format="channels_first")
        self.conv2d_transp_2 = Conv2DTranspose(4, 6, strides=2, activation="sigmoid", data_format="channels_first")

    def call(self, inputs):
        x = self.repeat(inputs)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.time_distr_dense(x)
        x = self.flatten(x)
        x = self.reshape(x)
        x = self.conv2d_transp_1(x)
        return self.conv2d_transp_2(x)


class Text2Image(tf.keras.Model):
    """ Text to image model using an encoder-decoder structure
        composed of LSTM and Conv2DTranspose layers.

        Given a sequence of input characters representing
        an addition/subtraction between two numbers of maximum 3 digits,
        it outputs the result as a sequence of 4 28x28 images (sign and digits).
    """
    
    def __init__(self, weights_path=None):
        """ weights_path: path to the pretrained weights
        """
        super().__init__()
        self.weights_path = weights_path
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, inputs):
        embedding = self.encoder(inputs)
        return self.decoder(embedding)

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(max_query_length, len(unique_characters)))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def train(self, x_train, x_test, y_train, y_test, weights_path="pretrained_weights"):
        self.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["mae"])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0001, patience=20)
        history = self.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.05, callbacks=[early_stop])
        t2i.evaluate(x_test, y_test)
        t2i.save_weights(weights_path)
        return history


def encode_text(text):
        """ OneHot encode the input text into a 7x13 matrix
            Each row represents a char
        """
        char_map = dict(zip(unique_characters, range(len(unique_characters))))
        one_hot_mat = np.zeros((1, max_query_length, len(unique_characters)))
        for i, char in enumerate(text):
            one_hot_mat[0, i, char_map[char]] = 1
        return one_hot_mat


def save_output(output, output_path, img_name):
    if output_path[-1] not in ["/", "\\"]:
        img_name = "/" + img_name

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(output[0][i])
        plt.axis("off")
    plt.savefig(output_path + img_name + ".png")
    print(f"Output saved in {output_path}{img_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument("--train", help="Include this argument to train the model", action="store_true")
    parser.add_argument("--eval", type=str, default=None, help="String to evaluate. \
        Must have the form ddd?ddd, d=digit or whitespace and ?=\"+\" or \"-\"")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--summary", help="Include this argument to print the summary of the model", action="store_true")
    args = parser.parse_args()

    if not args.train and args.eval is None:
        parser.error("Use at least one argument between --train and --eval")

    # instantiate t2i model and create weights
    t2i = Text2Image()
    t2i(tf.ones(shape=(1, max_query_length, len(unique_characters))))

    if args.summary:
        t2i.build_graph().summary()

    # load dataset and onehot encode it
    x_text, y_img = np.load("../../MSc/IntroDL/Assign3/X_text.npy"), np.load("../../MSc/IntroDL/Assign3/y_img.npy")
    x_text_oh = np.zeros((x_text.shape[0], max_query_length, len(unique_characters)))
    for i, text in enumerate(x_text):
        x_text_oh[i] = encode_text(text)

    if args.pretrained is not None:
        t2i.load_weights(args.pretrained)

    if args.train:
        # start training
        x_train, x_test, y_train, y_test = train_test_split(x_text_oh, y_img, test_size=0.6, shuffle=True, random_state=42)
        t2i.train(x_train, x_test, y_train, y_test)

    if args.eval is not None:
        # evaluate a single string given as argument --eval and save the output as a png
        # check if the string has the correct format
        err_str = "Evaluation string must have the form ddd?ddd, d=digit or whitespace and ?=\"+\" or \"-\""
        if len(args.eval) != 7:
            exit(err_str)
        elif not re.search(r"(?:  | \d|\d\d)\d(?:\+|-)(?:  | \d|\d\d)\d", args.eval):
            exit(err_str)
        else:
            save_output(t2i(encode_text(args.eval)), output_path=".", img_name=args.eval.replace(" ", ""))