from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import main
import os
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Dense, Flatten
import keras.backend as K
from Sequence import DataSequence
import time
import wandb


class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.__network_path = main.NETWORK_NAME
        self.__model = None
        self.__encoder = None
        self.__decoder = None
        self.__batch_size = main.BATCH_SIZE  # Размер батча
        self.__img_shape = (main.IMG_SHAPE, main.IMG_SHAPE, 3)
        self.__datebase_name = main.DATEBASE_NAME  # Относительный путь до базы данных
        self.__q_train = main.Q_TRAIN  # Количество обучающих примеров
        self.__q_valid = main.Q_VALID  # Количество валидационных примеров
        self.__epochs = main.EPOCHS  # Кол-во эпох
        self.__optimizer = tf.keras.optimizers.Adam(1e-4)
        self.__latent_dim = 2


    def __call__(self, *args, **kwargs):
        return self.__model(*args, **kwargs)

    def create(self):
        """Создание модели"""
        input_img = Input(shape=self.__img_shape)
        x = Conv2D(64, 3, padding="same", activation="relu", strides=(2, 2))(input_img)
        x = Conv2D(128, 3, padding="same", activation="relu", strides=(2, 2))(x)
        x = Conv2D(256, 3, padding="same", activation="relu", strides=(2, 2))(x)
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        out_encoder = Dense(256, activation="relu")(x)

        decoder_input = tf.keras.layers.Input(shape=(128,))
        x = Dense(np.prod(shape_before_flattening[1:]),
                                  activation="relu")(decoder_input)

        x = Reshape(shape_before_flattening[1:])(x)
        x = Conv2DTranspose(256, 3, padding="same", activation="relu", strides=(2, 2))(x)
        x = Conv2DTranspose(128, 3, padding="same", activation="relu", strides=(2, 2))(x)
        x = Conv2DTranspose(64, 3, padding="same", activation="relu", strides=(2, 2))(x)
        x = Conv2D(3, 3, padding='same', activation="sigmoid")(x)

        self.__encoder = tf.keras.Model(input_img, out_encoder)
        self.__decoder = tf.keras.Model(decoder_input, x)
        self.__encoder.summary()
        self.__decoder.summary()
        print("Модель создана")


    def save(self):
        """Сохранение модели"""
        self.__encoder.save('enc_'+self.__network_path)
        self.__decoder.save('dec_'+self.__network_path)
        print("Модель сохранена")

    def load(self):
        """Загрузка модели"""
        try:
            self.__encoder = tf.keras.models.load_model('enc_'+self.__network_path)
            self.__decoder = tf.keras.models.load_model('dec_'+self.__network_path)

            print(f"Moдель {self.__network_path} загружена")
        except:
            print(f"Moдель {self.__network_path} не найдена")

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.__latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.__encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.__decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def train(self, sp=False, SNR=0.5):
        """Обучение модели"""
        train_dataset = DataSequence([os.path.join(self.__datebase_name, filename)
                        for filename in os.listdir(self.__datebase_name)][:self.__q_train])
        test_dataset = DataSequence([os.path.join(self.__datebase_name, filename)
                            for filename in os.listdir(self.__datebase_name)]
                                     [self.__q_train: self.__q_train + self.__q_valid])

        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(
                -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                axis=raxis)

        def compute_loss(model, x):
            mean, logvar = model.encode(x)
            z = model.reparameterize(mean, logvar)
            x_logit = model.decode(z)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
            logpz = log_normal_pdf(z, 0., 0.)
            logqz_x = log_normal_pdf(z, mean, logvar)
            return -tf.reduce_mean(logpx_z + logpz - logqz_x)

        @tf.function
        def train_step(model, x, optimizer):
            """Executes one training step and returns the loss.

            This function computes the loss and gradients, and uses the latter to
            update the model's parameters.
            """
            with tf.GradientTape() as tape:
                loss = compute_loss(model, x)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epochs = self.__epochs
        # set the dimensionality of the latent space to a plane for visualization later
        num_examples_to_generate = 16

        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        random_vector_for_generation = tf.random.normal(
            shape=[num_examples_to_generate, self.__latent_dim])


        # Pick a sample of the test set for generating output images
        assert self.__batch_size >= num_examples_to_generate
        for test_batch in test_dataset:
            test_sample = test_batch[0:num_examples_to_generate, :, :, :]


        wandb.init(project="VAE")
        self.generate_and_save_images(0, test_sample)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x in train_dataset:
                train_step(self, train_x, self.__optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(self, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch, elbo, end_time - start_time))
            wandb.log({"Test set ELBO": elbo, "time elapse for current epoch": end_time - start_time})
            self.generate_and_save_images(epoch, test_sample)

        wandb.finish()
        print("Модель обучена")
        self.save()

    def generate_and_save_images(self, epoch, test_sample):
        mean, logvar = self.encode(test_sample)
        z = self.reparameterize(mean, logvar)
        predictions = self.sample(z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0])
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

        #plt.show()




