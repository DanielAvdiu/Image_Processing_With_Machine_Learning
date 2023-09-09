import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from skimage import io

datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

x = io.imread('./augmentedData/masks/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.jpg')
x = x.reshape((1, ) + x.shape())
print(x.shape)


i=0

for batch in datagen.flow(x, batch_size=16,
                        save_to_dir='./augmentedData/masks',
                        save_prefix='aug',
                        save_format='jpg'
                          ):
    i+=1
    if i>20:
        break
