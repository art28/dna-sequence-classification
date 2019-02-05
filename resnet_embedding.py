import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from blocks_resnet import IdentitiyBlock_3, ConvolutionBlock_3
from tqdm import tqdm, tqdm_notebook
import time
from colorama import Fore, Style
import math


# https://github.com/tensorflow/models/blob/master/official/transformer/model/model_utils.py
def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position
    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


class ResnetEmbedding(tf.keras.Model):
    def __init__(self,
                 input_dim=(60),
                 num_words=8,
                 hidden_dim=128,
                 out_dim=3,
                 learning_rate=1e-3,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(ResnetEmbedding, self).__init__()
        self.input_dim = input_dim
        self.num_words = num_words
        self.hidden_dim=hidden_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        # layer declaration
        self.embedding = tf.keras.layers.Embedding(self.num_words, self.hidden_dim)

        self.conv1 = tf.layers.Conv1D(filters=64, kernel_size=7, strides=1, padding="same",
                                      activation=tf.nn.relu)
        self.maxpool1 = tf.layers.MaxPooling1D(3, 1, padding="same")

        self.conv2a = ConvolutionBlock_3(filters=[32, 32, 64], kernel_sizes=[1,3,1,1])
        self.iden2b = IdentitiyBlock_3([32, 32, 64], [1,3,1])
        self.iden2c = IdentitiyBlock_3([32, 32, 64], [1,3,1])

        self.conv3a = ConvolutionBlock_3(filters=[64, 64, 128], kernel_sizes=[1,3,1,1])
        self.iden3b = IdentitiyBlock_3([64, 64, 128], [1,3,1])
        self.iden3c = IdentitiyBlock_3([64, 64, 128], [1,3,1])
        self.iden3d = IdentitiyBlock_3([64, 64, 128], [1,3,1])

        self.conv4a = ConvolutionBlock_3(filters=[128, 128, 512], kernel_sizes=[1,3,1,1])
        self.iden4b = IdentitiyBlock_3([128, 128, 512], [1,3,1])
        self.iden4c = IdentitiyBlock_3([128, 128, 512], [1,3,1])
        self.iden4d = IdentitiyBlock_3([128, 128, 512], [1,3,1])
        self.iden4e = IdentitiyBlock_3([128, 128, 512], [1,3,1])
        self.iden4f = IdentitiyBlock_3([128, 128, 512], [1,3,1])

        self.conv5a = ConvolutionBlock_3(filters=[256, 256, 1024], kernel_sizes=[1,3,1,1])
        self.iden5b = IdentitiyBlock_3([256, 256, 1024], [1,3,1])
        self.iden5c = IdentitiyBlock_3([256, 256, 1024], [1,3,1])

        self.flatten = tf.layers.Flatten()
        self.dropout = tf.layers.Dropout(0.5)
        self.out_layer = tf.layers.Dense(out_dim)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # global step
        self.global_step = 0

    def build(self):
        dummy_input = tf.constant(tf.zeros((1,) + self.input_dim))
        dummy_pred = self.call(dummy_input, True)
        self.built = True

    def call(self, X, training):
        """predicting output of the network
        Args:
            X : input tensor
            training : whether apply dropout or not
        """
        x = self.embedding(X)
        x += get_position_encoding(tf.shape(x)[1], self.hidden_dim)

        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2a(x, training=training)
        x = self.iden2b(x, training=training)
        x = self.iden2c(x, training=training)

        x = self.conv3a(x, training=training)
        x = self.iden3b(x, training=training)
        x = self.iden3c(x, training=training)
        x = self.iden3d(x, training=training)

        x = self.conv4a(x, training=training)
        x = self.iden4b(x, training=training)
        x = self.iden4c(x, training=training)
        x = self.iden4d(x, training=training)
        x = self.iden4e(x, training=training)
        x = self.iden4f(x, training=training)

        x = self.conv5a(x, training=training)
        x = self.iden5b(x, training=training)
        x = self.iden5c(x, training=training)

        if training:
            x = self.dropout(x)

        x = self.out_layer(self.flatten(x))

        return x

    def loss(self, X, y, training):
        """calculate loss of the batch
        Args:
            X : input tensor
            y : target label(class number)
            training : whether apply dropout or not
        """
        prediction = self.call(X, training)
        loss_value = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)
        return loss_value, prediction

    def grad(self, X, y, trainig):
        """calculate gradient of the batch
        Args:
            X : input tensor
            y : target label(class number)
            training : whether apply dropout or not
        """
        with tfe.GradientTape() as tape:
            loss_value, prediction = self.loss(X, y, trainig)
        return tape.gradient(loss_value, self.variables), loss_value, prediction

    def fit(self, X_train, y_train, X_val, y_val, epochs=1, verbose=1, batch_size=32, saving=False, tqdm_option=None):
        """train the network
        Args:
            X_train : train dataset input
            y_train : train dataset label
            X_val : validation dataset input
            y_val = validation dataset input
            epochs : training epochs
            verbose : for which step it will print the loss and accuracy (and saving)
            batch_size : training batch size
            saving: whether to save checkpoint or not
            tqdm_option: tqdm logger option. default is none. use "normal" for tqdm, "notebook" for tqdm_notebook
        """

        def tqdm_wrapper(*args, **kwargs):
            if tqdm_option == "normal":
                return tqdm(*args, **kwargs)
            elif tqdm_option == "notebook":
                return tqdm_notebook(*args, **kwargs)
            else:
                return args[0]

        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(999999999).batch(batch_size)
        batchlen_train = (len(X_train) - 1) // batch_size + 1

        dataset_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(999999999).batch(batch_size)
        batchlen_val = (len(X_val) - 1) // batch_size + 1

        with tf.device(self.device_name):
            for i in range(epochs):
                epoch_loss = 0.0
                self.global_step += 1
                train_accuracy = tf.contrib.eager.metrics.Accuracy()
                for X, y in tqdm_wrapper(dataset_train, total=batchlen_train, desc="GLOBAL %s" % self.global_step):
                    grads, batch_loss, pred = self.grad(X, y, True)
                    mean_loss = tf.reduce_mean(batch_loss)
                    epoch_loss += mean_loss
                    self.optimizer.apply_gradients(zip(grads, self.variables))
                    train_accuracy(tf.argmax(pred, axis=1), y)

                epoch_loss_val = 0.0
                val_accuracy = tf.contrib.eager.metrics.Accuracy()
                for X, y in tqdm_wrapper(dataset_val, total=batchlen_val, desc="GLOBAL %s" % self.global_step):
                    batch_loss, pred = self.loss(X, y, False)
                    epoch_loss_val += tf.reduce_mean(batch_loss)
                    val_accuracy(tf.argmax(pred, axis=1), y)

                if i == 0 or ((i + 1) % verbose == 0):
                    print(Fore.RED + "=" * 25)
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % (epoch_loss / batchlen_train))
                    print("VAL   loss   : %.4f" % (epoch_loss_val / batchlen_val))
                    print("TRAIN acc    : %.4f%%" % (train_accuracy.result().numpy() * 100))
                    print("VAL   acc    : %.4f%%" % (val_accuracy.result().numpy() * 100))

                    if saving:
                        self.save()
                    print("=" * 25 + Style.RESET_ALL)
                    time.sleep(1)

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)
        print("saved step %d in %s" % (self.global_step, self.checkpoint_directory))

    def load(self, global_step="latest"):
        self.build()

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = global_step
