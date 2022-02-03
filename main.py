import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Layer, Dense, LSTM, Flatten, TimeDistributed
from tensorflow.keras.layers import RepeatVector, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam


max_query_length = 7
max_answer_length = 4
unique_characters = "0123456789+- "


class Encoder(Layer):
    """Description"""

    def __init__(self, latent_dim=128):
        """ latent_dim: dimension of the space where the input text is encoded to
        """
        super().__init__()
        self.latent_dim = latent_dim

    def call(self, inputs):
        x = LSTM(self.latent_dim, return_sequences=True)(inputs)
        x = LSTM(self.latent_dim, return_sequences=True)(x)
        return LSTM(self.latent_dim)(x)


class Decoder(Layer):
    """Description"""

    def call(self, inputs):
        x = RepeatVector(4)(inputs)
        x = LSTM(256, return_sequences=True)(x)
        x = LSTM(256, return_sequences=True)(x)
        x = TimeDistributed(Dense(1024, activation="relu"))(x)
        x = Flatten()(x)
        x = Reshape((64, 8, 8))(x)
        x = Conv2DTranspose(8, 5, strides=1, data_format="channels_first")(x)
        return Conv2DTranspose(4, 6, strides=2, activation="sigmoid", data_format="channels_first")(x)


class Text2Image(tf.keras.Model):
    """Description"""
    
    def __init__(self, weights_path=None):
        """ weights_path: path to the pretrained weights
        """
        super().__init__()
        self.weights_path = weights_path
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, inputs):
        # oh_input = self.encode_text(inputs)
        embedding = self.encoder(inputs)
        return self.decoder(embedding)

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(max_query_length, len(unique_characters)))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


def encode_text(text):
        """ OneHot encode the input text into a 7x13 matrix
            Each row represents a char
        """
        char_map = dict(zip(unique_characters, range(len(unique_characters))))
        one_hot_mat = np.zeros((1, max_query_length, len(unique_characters)))
        for i, char in enumerate(text):
            one_hot_mat[0, i, char_map[char]] = 1
        return one_hot_mat


def save_output(output, output_path):
    img_name = "output.png"
    if output_path[-1] not in ["/", "\\"]:
        img_name = "/" + img_name

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(output[0][i])
        plt.axis("off")
    plt.savefig(output_path + img_name)
    print(f"Output saved in {output_path}{img_name}")


if __name__ == "__main__":
    t2i = Text2Image("pretrained_weights/t2i")
    t2i.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    t2i(tf.ones(shape=(1, max_query_length, len(unique_characters))))
    # t2i.build_graph().summary()
    t2i.load_weights(t2i.weights_path)
    save_output(t2i(encode_text("1+1")), output_path=".")