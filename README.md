# Sputnik_VAE
Sputnik Task 2
В данном програмном обеспечении реализован вариационный автоэнкодер.
Архитектура модели представляет собой слои:

	знкодер:
 
	Conv2D(64, 3, padding="same", activation="relu", strides=(2, 2))
        Conv2D(128, 3, padding="same", activation="relu", strides=(2, 2))
        Conv2D(256, 3, padding="same", activation="relu", strides=(2, 2))
		shape_before_flattening = K.int_shape(x)
        Flatten()
        Dense(512, activation="relu")(x)
	
	декодер:

        decoder_input = tf.keras.layers.Input(shape=(256,))
		Dense(np.prod(shape_before_flattening[1:]))
		Reshape(shape_before_flattening[1:])(x)
		Conv2DTranspose(256, 3, padding="same", activation="relu", strides=(2, 2))(x)
        Conv2DTranspose(128, 3, padding="same", activation="relu", strides=(2, 2))(x)
        Conv2DTranspose(64, 3, padding="same", activation="relu", strides=(2, 2))(x)
        Conv2D(3, 3, padding='same', activation="sigmoid")(x)
		

В качестве дата сета выбраны первые 20000 изображений img_align_celeba
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
На входе размер изображения изменяется до 244*244
после завершения программы создаётся GIF, который наглядно показывает, как изображение по эпохам

