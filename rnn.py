import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from tqdm import tqdm, tqdm_notebook
import time
from colorama import Fore, Style
from resnet_embedding import get_position_encoding


class LSTM(tf.keras.Model):
    def __init__(self,
                 input_dim=(60),
                 num_words=8,
                 hidden_dim=128,
                 out_dim=3,
                 learning_rate=1e-3,
                 checkpoint_directory="checkpoints/",
                 use_cudnn=False,
                 device_name="cpu:0"):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.num_words = num_words
        self.hidden_dim=hidden_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.use_cudnn = use_cudnn
        self.device_name = device_name

        # layer declaration
        self.embedding = tf.keras.layers.Embedding(self.num_words, self.hidden_dim)
        if use_cudnn:
            self.lstm = tf.keras.layers.CuDNNLSTM(512)
        else:
            self.lstm = tf.keras.layers.LSTM(512)
        self.dense2 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.dense3 = tf.layers.Dense(128, activation=tf.nn.relu)
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

        x = self.lstm(x)
        x = self.dense2(x)
        x = self.dense3(x)

        if training:
            x = self.dropout(x)

        x = self.out_layer(x)

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
        loss_value = tf.reduce_mean(loss_value)
        l2loss = tf.nn.l2_loss(self.out_layer.weights[0])
        loss_value += l2loss
        return loss_value, l2loss, prediction

    def grad(self, X, y, trainig):
        """calculate gradient of the batch
        Args:
            X : input tensor
            y : target label(class number)
            training : whether apply dropout or not
        """
        with tfe.GradientTape() as tape:
            loss_value, l2loss, prediction = self.loss(X, y, trainig)
        return tape.gradient(loss_value, self.variables), loss_value, l2loss, prediction

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
                epoch_l2loss = 0.0
                self.global_step += 1
                train_accuracy = tf.contrib.eager.metrics.Accuracy()
                for X, y in tqdm_wrapper(dataset_train, total=batchlen_train, desc="GLOBAL %s" % self.global_step):
                    grads, batch_loss, l2loss, pred = self.grad(X, y, True)
                    mean_loss = tf.reduce_mean(batch_loss)
                    epoch_loss += mean_loss
                    epoch_l2loss += l2loss
                    self.optimizer.apply_gradients(zip(grads, self.variables))
                    train_accuracy(tf.argmax(pred, axis=1), y)

                epoch_loss_val = 0.0
                epoch_l2loss_val = 0.0
                val_accuracy = tf.contrib.eager.metrics.Accuracy()
                for X, y in tqdm_wrapper(dataset_val, total=batchlen_val, desc="GLOBAL %s" % self.global_step):
                    batch_loss, l2loss, pred = self.loss(X, y, False)
                    epoch_loss_val += tf.reduce_mean(batch_loss)
                    epoch_l2loss_val += l2loss
                    val_accuracy(tf.argmax(pred, axis=1), y)

                if i == 0 or ((i + 1) % verbose == 0):
                    print(Fore.RED + "=" * 25)
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss     : %.4f" % (epoch_loss / batchlen_train))
                    print(Fore.BLUE + "TRAIN loss[l2] : %.4f" % (epoch_l2loss / batchlen_train))
                    print(Fore.GREEN + "TRAIN loss[net]: %.4f" % (
                                epoch_loss / batchlen_train - epoch_l2loss / batchlen_train))
                    print(Fore.RED + "VAL   loss     : %.4f" % (epoch_loss_val / batchlen_val))
                    print(Fore.BLUE + "VAL   loss[l2] : %.4f" % (epoch_l2loss_val / batchlen_val))
                    print(Fore.GREEN + "TRAIN loss[net]: %.4f" % (
                                epoch_loss_val / batchlen_val - epoch_l2loss_val / batchlen_val))
                    print(Fore.RED + "TRAIN acc      : %.4f%%" % (train_accuracy.result().numpy() * 100))
                    print("VAL   acc      : %.4f%%" % (val_accuracy.result().numpy() * 100))

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
