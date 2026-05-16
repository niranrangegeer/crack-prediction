# 1、Import libraries
import os
import tensorflow as tf
import matplotlib.pyplot as plt
#from IPython import display
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = InteractiveSession(config=config)
import pandas as pd
from tensorflow.python.keras.utils.vis_utils import plot_model


import random

#  2、Data paths
PATH_train = './Train_images/'
PATH_test = './Test_images/'

# 3、load images
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 3
    input_image = image[:, :w, :]
    energy_image = image[:, w:2*w, :]
    real_image = image[:, 2*w:, :]

    input_image = tf.cast(input_image, tf.float32)
    energy_image = tf.cast(energy_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, energy_image, real_image

image_file=PATH_train+'Train_1.jpg'


# 4.1 Resize iamges
def resize(input_image, energy_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    energy_image = tf.image.resize(energy_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, energy_image, real_image

# 4.2 Random cropping
# traget image sizeing
IMG_WIDTH = 256
IMG_HEIGHT = 256
def random_crop(input_image, energy_image, real_image):
    stacked_image = tf.stack([input_image, energy_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[3, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1], cropped_image[2]

# 4.3 random mirroring
def random_jitter(input_image,energy_image, real_image):
    # resizing to 286 x 286 x 1
    input_image, energy_image, real_image = resize(input_image, energy_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 1
    input_image, energy_image, real_image = random_crop(input_image,energy_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        energy_image = tf.image.flip_left_right(energy_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, energy_image, real_image

# 4.4 normalizing the images to [-1, 1]
def normalize(input_image, energy_image, real_image):
    input_image = (input_image / 127.5) - 1
    energy_image = (energy_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, energy_image, real_image

# 4.5 Loading training images
def load_image_train(image_file):
    input_image, energy_image, real_image = load(image_file)
    input_image, energy_image, real_image = random_jitter(input_image, energy_image, real_image)
    input_image, energy_image, real_image = normalize(input_image, energy_image, real_image)

    return input_image, energy_image, real_image

# 4.6 Loading test images
def load_image_test(image_file):
    input_image, energy_image, real_image = load(image_file)
    input_image, energy_image, real_image = resize(input_image, energy_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, energy_image, real_image = normalize(input_image, energy_image, real_image)

    return input_image, energy_image, real_image


# 4.7 Preparing training dataset
BUFFER_SIZE = 120
BATCH_SIZE = 16

train_dataset = tf.data.Dataset.list_files(PATH_train+'Train_*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# 4.8 Preparing training dataset
test_dataset = tf.data.Dataset.list_files(PATH_test+'Test_*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


# 5.1 Define the Downsampling function

OUTPUT_CHANNELS = 3
def downsample(filters, size, stride, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
          tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                                 kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

# 5.2 Define the Upsampling function
def upsample(filters, size, stride, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result

# Maxpool
def downpool(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
          tf.keras.layers.MaxPooling2D(pool_size = (size, size)))
    result.add(
          tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

# 5.2 Full-scale skip connections
def upsample2(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.UpSampling2D(size=(size, size), interpolation='bilinear'))

    result.add(
          tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def conv(filters):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
          tf.keras.layers.Conv2D(filters*5, 3, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

# def Generatoor
def Generator():

    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 3], name='input_image')

    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)
    xe1 = inp
    xe2 = downsample(32, 4, 2, apply_batchnorm=False)(xe1) #128
    xe3 = downsample(64, 4, 2)(xe2) #64
    xe4 = downsample(128, 4, 2)(xe3) #32
    xe5 = downsample(128, 4, 2)(xe4)    #16
    xe6 = downsample(256, 4, 2)(xe5)    #8
    xe7 = downsample(256, 4, 2)(xe6)    #4
    xe8 = downsample(512, 4, 2)(xe7)    #2
    xe9 = downsample(512, 4, 2)(xe8)    #1

    fiter = 32
    
    x81 = downpool(fiter, 128)(xe1)
    x82 = downpool(fiter, 64)(xe2)
    x83 = downpool(fiter, 32)(xe3)
    x84 = downpool(fiter, 16)(xe4)
    x85 = downpool(fiter, 8)(xe5)
    x86 = downpool(fiter, 4)(xe6)
    x87 = downpool(fiter, 2)(xe7)
    x88 = downsample(fiter, 3, 1)(xe8)
    x89 = upsample(fiter, 4, 2, apply_dropout=True)(xe9)
    xd8 = tf.keras.layers.Concatenate()([x89, x88, x87, x86, x85, x84, x83, x82, x81]) #2
    xd8 = conv(288)(xd8)

    x71 = downpool(fiter, 64)(xe1)
    x72 = downpool(fiter, 32)(xe2)
    x73 = downpool(fiter, 16)(xe3)
    x74 = downpool(fiter, 8)(xe4)
    x75 = downpool(fiter, 4)(xe5)
    x76 = downpool(fiter, 2)(xe6)
    x77 = downsample(fiter, 3, 1)(xe7)
    x78 = upsample(fiter, 4, 2, apply_dropout=True)(xd8)
    x79 = upsample2(fiter, 4)(xe9)
    xd7 = tf.keras.layers.Concatenate()([x79, x78, x77, x76, x75, x74, x73, x72, x71]) #4
    xd7 = conv(288)(xd7)
    
    
    x61 = downpool(fiter, 32)(xe1)
    x62 = downpool(fiter, 16)(xe2)
    x63 = downpool(fiter, 8)(xe3)
    x64 = downpool(fiter, 4)(xe4)
    x65 = downpool(fiter, 2)(xe5)
    x66 = downsample(fiter, 3, 1)(xe6)
    x67 = upsample(fiter, 4, 2, apply_dropout=True)(xd7)
    x68 = upsample2(fiter, 4)(xd8)
    x69 = upsample2(fiter, 8)(xe9)
    xd6 = tf.keras.layers.Concatenate()([x69, x68, x67, x66, x65, x64, x63, x62, x61]) #8
    xd6 = conv(288)(xd6)
    
    
    x51 = downpool(fiter, 16)(xe1)
    x52 = downpool(fiter, 8)(xe2)
    x53 = downpool(fiter, 4)(xe3)
    x54 = downpool(fiter, 2)(xe4)
    x55 = downsample(fiter, 3, 1)(xe5)
    x56 = upsample(fiter, 4, 2, apply_dropout=True)(xd6)
    x57 = upsample2(fiter, 4)(xd7)
    x58 = upsample2(fiter, 8)(xd8)
    x59 = upsample2(fiter, 16)(xe9)
    xd5 = tf.keras.layers.Concatenate()([x59, x58, x57, x56, x55, x54, x53, x52, x51]) #16
    xd5 = conv(288)(xd5)

    x41 = downpool(fiter, 8)(xe1)
    x42 = downpool(fiter, 4)(xe2)
    x43 = downpool(fiter, 2)(xe3)
    x44 =downsample(fiter, 3, 1)(xe4)
    x45 = upsample(fiter, 4, 2)(xd5)
    x46 = upsample2(fiter, 4)(xd6)
    x47 = upsample2(fiter, 8)(xd7)
    x48 = upsample2(fiter, 16)(xd8)
    x49 = upsample2(fiter, 32)(xe9)
    xd4 = tf.keras.layers.Concatenate()([x49, x48, x47, x46, x45, x44, x43, x42, x41]) #32
    xd4 = conv(288)(xd4)

    x31 = downpool(fiter, 4)(xe1)
    x32 = downpool(fiter, 2)(xe2)
    x33 = downsample(fiter, 3, 1)(xe3)
    x34 = upsample(fiter, 4, 2)(xd4)
    x35 = upsample2(fiter, 4)(xd5)
    x36 = upsample2(fiter, 8)(xd6)
    x37 =  upsample2(fiter, 16)(xd7)
    x38 = upsample2(fiter, 32)(xd8)
    x39 = upsample2(fiter, 64)(xe9)
    xd3 = tf.keras.layers.Concatenate()([x39, x38, x37, x36, x35, x34, x33, x32, x31]) #64
    xd3 = conv(288)(xd3)

    x21 = downpool(fiter, 2)(xe1)
    x22 = downsample(fiter, 3, 1)(xe2)
    x23 = upsample(fiter, 4, 2)(xd3)
    x24 = upsample2(fiter, 4)(xd4)
    x25 = upsample2(fiter, 8)(xd5)
    x26 = upsample2(fiter, 16)(xe6)
    x27 = upsample2(fiter, 32)(xd7)
    x28 = upsample2(fiter, 64)(xd8)
    x29 = upsample2(fiter, 128)(xe9)
    xd2 = tf.keras.layers.Concatenate()([x29, x28, x27, x26, x25, x24, x23, x22, x21]) #128
    xd2 = conv(288)(xd2)

    x11 = downsample(fiter, 3, 1)(xe1)
    x12 = upsample(fiter, 4, 2)(xd2)
    x13 = upsample2(fiter, 4)(xd3)
    x14 = upsample2(fiter, 8)(xd4)
    x15 = upsample2(fiter, 16)(xd5)
    x16 = upsample2(fiter, 32)(xe6)
    x17 = upsample2(fiter, 64)(xd7)
    x18 = upsample2(fiter, 128)(xd8)
    x19 = upsample2(fiter, 256)(xe9)
    xd1 = tf.keras.layers.Concatenate()([x19, x18, x17, x16, x15, x14, x13, x12, x11]) #256
    xd1 = conv(288)(xd1)



    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=1,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)
    x = last(xd1)
    return tf.keras.Model(inputs=inp, outputs=x)
generator = Generator()

# 5.4 Print Generator
tf.keras.utils.plot_model(generator, to_file=r'./Generator.png',show_shapes=True, dpi=64)

# 5.5 Def Discriminator（PatchGAN）
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 3], name='input_image1')
    tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, 2, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4, 2)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4, 2)(down2) # (bs, 32, 32, 256)


    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
# 5.6 Print Discriminator
tf.keras.utils.plot_model(discriminator, './Discriminator.png' ,show_shapes=True, dpi=64)

# 6.1 def Generator target loss function
LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # MAE
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + LAMBDA*l1_loss


    return total_gen_loss, gan_loss, l1_loss, l1_loss

#6.2 Def Discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss, real_loss


# 7 Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)

# 8 Image generation and display
def generate_images(model, test_input, test_energy, tar, epoch,kj):
    prediction = model([test_input, test_energy], training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_energy[0],  tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.

        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('./Out_rock/Test700_'+str(epoch)+'_'+str(kj)+'.png', transparent=True)
    plt.close('all')
    
# 9 Gradient descent process
def train_step(input_image,energy_image, target, epoch, n):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([input_image, energy_image], training=True)

        disc_real_output = discriminator([input_image, energy_image, target], training=True)
        disc_generated_output = discriminator([input_image, energy_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss ,gen_Dice_loss= generator_loss(disc_generated_output, gen_output, target)
        disc_loss ,disc_real_loss= discriminator_loss(disc_real_output, disc_generated_output)

        if n==99:
            f= open('Loss.txt','a')
            f.write(str(epoch)+' '+str(n.numpy())+' '+str(gen_l1_loss.numpy())+' '+str(gen_Dice_loss.numpy())
                    +' '+str(gen_gan_loss.numpy())+' '+str(disc_loss.numpy())+' '+str(disc_real_loss.numpy())+'\n')
            f.close()

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
generator = tf.saved_model.load('./SaveModel/generator_600')
discriminator = tf.saved_model.load('./SaveModel/discriminator_100')

# 10.1 Defining the training process
def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        display.clear_output(wait=True)
        if (epoch+1) % 100 == 0:
            kj=1
            for example_input, example_energy, example_target in test_ds.take(10):
                generate_images(generator, example_input, example_energy, example_target, epoch+1,kj)
                kj=kj+1

        print("Epoch: ", epoch)        
            
        for n, (input_image, energy_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 50 == 0:
                print()
            train_step(input_image, energy_image, target, epoch, n)
        print()
        
# 10.2 strat train
print('strat train')
EPOCHS = 1001
fit(train_dataset, EPOCHS, test_dataset)
