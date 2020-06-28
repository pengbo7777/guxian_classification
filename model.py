import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import os

batch_size = 10

num_classes = 2
epochs = 100
validation_ratio = 0.2

sampling_type = "long"
data_path = "D:\proj-doc\成分分析\\6-16相机采样\long"

data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cnn_trained_model.h5'

model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(512, 512, 3)))
# 通道数可能变
model.add(Activation('relu'))
model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=1e-4, decay=1e-6)
# lr可能也要调

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def my_random_crop(image):
    ret = []
    for i in range(len(image)):
        y = int(np.random.randint(1080 - 512 + 1))
        x = int(np.random.randint(1920 - 512 + 1))
        h = 512
        w = 512
        image_crop = image[i, y:y + h, x:x + w, :]
        if image_crop.shape[0] != 512 or image_crop.shape[1] != 512:
            print('image size error')
        ret.append(image_crop)
    return np.array(ret)


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=validation_ratio)
    # 配参数


def load_data(data_path, validation_ratio):
    datagen = ImageDataGenerator()
    data_iter = datagen.flow_from_directory(data_path,
                                           batch_size=batch_size,
                                           class_mode="categorical",
                                           shuffle=True,
                                           target_size=(1080, 1920))

    iter_batch_size = len(data_iter)
    test_batch_size = int(validation_ratio * iter_batch_size)
    train_batch_size = iter_batch_size - test_batch_size

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    model = load_model("D:\proj-doc\成分分析\\6-15长纤短纤二分类cnn\src\saved_models\keras_cnn_trained_model.h5")
    for i in range(epochs):
        # train
        for j in range(train_batch_size):
            print(u"\r", 'epoch', i, 'batch', j, end="")
            x, y = it.next()
            x = my_random_crop(x)
            for k in range(len(x)):
                x[k] = datagen.random_transform(x[k])
            model.train_on_batch(x, y)
        # test per epoch
        for j in range(test_batch_size):
            x, y = it.next()
            x = my_random_crop(x)
            scores = model.evaluate(x, y, verbose=1)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().


#    model.fit_generator(datagen.flow_from_directory(data_path,
#                                     batch_size=batch_size,
#                                     class_mode="binary",
#                                     shuffle=True,
#                                     target_size=(1080, 1920)),
#                        epochs=epochs,
#                        workers=4)

# 模型已经训练好了... 然后你可以调用model.predict看看 但是送进去的数据 需要是512 512 3的数据 而且范围要被调节到0-1之间(可能)

# simpledatagen = ImageDataGenerator()
# it = simpledatagen.flow_from_directory(data_path,
#                                  batch_size=batch_size,
#                                  class_mode="categorical",
#                                  shuffle=True,
#                                  target_size=(1080, 1920))

# x, y = it.next()
# x = my_random_crop(x)
# a = np.array(x)
# print(x.shape)
# print(y)
# model.predict(a)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# 加载模型
# x_test = np.array(x_test)
# y_test = np.array(y_test)
# model = load_model("D:\proj-doc\成分分析\\6-15长纤短纤二分类cnn\src\saved_models\keras_cnn_trained_model.h5")
# print(x_test.shape, y_test.shape)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

if __name__ == '__main__':
    simpledatagen = ImageDataGenerator()
    it = simpledatagen.flow_from_directory(data_path,
                                           batch_size=batch_size,
                                           class_mode="categorical",
                                           shuffle=True,
                                           target_size=(1080, 1920))
    print(it.__len__())
    print(len(it))
