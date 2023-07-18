import neural_network
from time import perf_counter
import imageio
import matplotlib.pyplot as plt
import glob


"""Глобальные константы"""
Q_TRAIN = 10000                   # Количество обучающих примеров
Q_VALID = 1000                   # Количество обучающих примеров
DATEBASE_NAME = 'img_align_celeba'          # Имя базы данных для обучения
BATCH_SIZE = 100                   # количество тренировочных изображений для обработки перед обновлением параметров модели
IMG_SHAPE = 224           # размерность к которой будет приведено входное изображение
NETWORK_NAME = "VAE.keras"             # Имя нейросети для загрузки и сохранения
EPOCHS = 10                     # Количество эпох


if __name__ == '__main__':
    start = perf_counter()
    """Основной раздел программы"""
    model = neural_network.NeuralNetwork()
    print(model.trainable_variables)
    """Создание и обучение нейросети. Закомментировать, если модель обучена"""
    """Аргумент sp выставляется для наложения фильтра соль/перец на входные изображения. SNR- процент шума"""
    model.create()
    model.train()
    print(f"Время на обучение модели {perf_counter() - start}")

    """Загрузка обученной модели"""
    #model.load()



    plt.axis('off')  # Display images

    anim_file = 'cvae.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)

    import tensorflow_docs.vis.embed as embed

    embed.embed_file(anim_file)


