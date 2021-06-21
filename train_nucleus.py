# STEP 1 TRAIN nucleus
import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf
import random
import imageio
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate


def create_train_and_val_df(path_to_imgfolder, num_of_slices, cutoff_first_slices, cutoff_last_slices, train_spheroid, val_spheroid, test_spheroid):
    img_path = Path(path_to_imgfolder)
    brightfield = [x.as_posix() for x in img_path.glob('*ch4*.png')]
    brightfield.sort()

    nucleus = [x.as_posix() for x in img_path.glob('*ch1*.png')]
    nucleus.sort()

    df_imgs = pd.DataFrame(data={'nucleus': nucleus, 'brightfield': brightfield})

    df_imgs['plane'] = df_imgs['brightfield'].apply(lambda x: int(x.split('-')[0].split('p')[-1]))
    df_imgs = df_imgs[(df_imgs.plane >= cutoff_first_slices)]
    df_imgs = df_imgs[(df_imgs.plane <= cutoff_last_slices)]
    df_imgs = df_imgs[(df_imgs.plane <= (df_imgs.plane.max() - (num_of_slices//2))) & (df_imgs.plane >= (df_imgs.plane.min() + (num_of_slices//2)))]
    
    df_imgs['number_of_slices'] = num_of_slices

    df_imgs = df_imgs.astype({col: 'int32' for col in df_imgs.select_dtypes('int64').columns})

    df_train = df_imgs[df_imgs['nucleus'].str.contains(train_spheroid)]
    df_val = df_imgs[df_imgs['nucleus'].str.contains(val_spheroid)]
    df_test = df_imgs[df_imgs['nucleus'].str.contains(test_spheroid)]

    return df_train, df_val, df_test

def read_imgs(nucleus, brightfield, plane, number_of_slices):
    start_plane = plane - int(number_of_slices//2)
    imgs = tf.TensorArray(tf.float32, size = number_of_slices)

    for i in range(number_of_slices):
        if i == 0:
            if start_plane < 10:
                fn = tf.strings.regex_replace(brightfield, tf.strings.format('p0{}', plane) if plane < 10 else tf.strings.format('p{}', plane), tf.strings.format('p0{}', start_plane))
            else:
                fn = tf.strings.regex_replace(brightfield, tf.strings.format('p0{}', plane) if plane < 10 else tf.strings.format('p{}', plane), tf.strings.format('p{}', start_plane))
        else:
            next_plane = start_plane + i
            if next_plane < 10:
                fn = tf.strings.regex_replace(brightfield, tf.strings.format('p0{}', plane) if plane < 10 else tf.strings.format('p{}', plane), tf.strings.format('p0{}', next_plane))
            else:
                fn = tf.strings.regex_replace(brightfield, tf.strings.format('p0{}', plane) if plane < 10 else tf.strings.format('p{}', plane), tf.strings.format('p{}', next_plane))

        img = tf.io.decode_png(tf.io.read_file(fn), channels = 1, dtype = tf.uint16) 
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_crop_or_pad(img, 1024, 1024)
        imgs = imgs.write(i, tf.squeeze(img))
    bfimg = imgs.stack()    
    bfimg = tf.transpose(bfimg, perm = [1,2,0])

    fluo_target = tf.io.decode_png(tf.io.read_file(nucleus), channels = 1, dtype = tf.uint16) 
    fluo_target = tf.image.convert_image_dtype(fluo_target, tf.float32)
    fluo_target = tf.image.resize_with_crop_or_pad(fluo_target, 1024, 1024)

    return bfimg, fluo_target

def create_unet_model(input_shape):
    def down(connected_layer, num_filters, pool = True):
        conv = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer = 'glorot_uniform')(connected_layer)
        conv = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer = 'glorot_uniform')(conv)

        if pool == True:
            pool = MaxPool2D(2)(conv)
            return conv, pool
        else:
            return conv

    def up(connected_layer, concat_layer, num_filters):
        up = UpSampling2D(2)(connected_layer)
        up = Conv2D(num_filters, 2, padding="same", activation = 'relu', kernel_initializer='glorot_uniform')(up)
        up = Concatenate(axis=3)([concat_layer, up])

        conv = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer='glorot_uniform')(up)
        conv = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer='glorot_uniform')(conv)
        return conv

    # "Down"
    unet_input = Input(input_shape)
    c1, p1 =  down(unet_input, 64)
    c2, p2 = down(p1, 128)
    c3, p3 = down(p2, 256)
    c4, p4 = down(p3, 512)
    c5 = down(p4, 1024, pool = False)
    # "Up"
    c6 = up(c5, c4, 512)
    c7 = up(c6, c3, 256)
    c8 = up(c7, c2, 128)
    c9 = up(c8, c1, 64)
    out = Conv2D(1, 1, activation=None)(c9)

    model = Model(inputs=unet_input, outputs = out, name = 'U-net')
    return model


def main():
    bs = 1
    cutoff_first_slices = 3
    cutoff_last_slices = 10

    train_spheroid = '|'.join(['r02c06', 'r02c07', 'r02c12', 'r02c15', 'r02c18', 'r02c19', 'r02c20', 'r02c22', 'r03c03', 'r03c08', 'r03c09', 'r03c10', 'r03c11', 'r03c12', 'r03c14', 'r03c15', 'r03c16', 'r03c19', 'r03c23', 'r04c04', 'r04c11', 'r04c12', 'r04c16', 'r05c08', 'r05c12', 'r05c13', 'r05c18', 'r05c20', 'r05c21', 'r05c23', 'r06c06', 'r06c07', 'r06c11', 'r06c15', 'r06c16', 'r06c17', 'r06c20', 'r07c09', 'r07c10', 'r07c11', 'r07c14', 'r07c15', 'r07c22', 'r08c09', 'r08c10', 'r08c13', 'r08c14', 'r08c15', 'r08c18', 'r08c19', 'r08c23', 'r09c07', 'r09c08', 'r09c09', 'r09c11', 'r09c12', 'r09c13', 'r09c15', 'r09c17', 'r09c21', 'r09c23', 'r10c08', 'r10c09', 'r10c10', 'r10c11', 'r10c16', 'r10c22', 'r11c08', 'r11c09', 'r11c12', 'r11c16', 'r11c18', 'r11c19', 'r12c09', 'r12c11', 'r12c12', 'r12c14', 'r12c17', 'r12c19', 'r13c06', 'r13c10', 'r13c12', 'r13c14', 'r13c17', 'r13c18', 'r13c19', 'r13c22', 'r14c14', 'r14c16', 'r14c21', 'r14c22', 'r15c12', 'r15c13', 'r15c16', 'r15c17', 'r15c20', 'r15c23'])
    val_spheroid = '|'.join(['r02c17', 'r03c13', 'r03c22', 'r04c08', 'r04c18', 'r04c20', 'r04c22', 'r05c10', 'r05c19', 'r06c14', 'r06c18', 'r06c19', 'r06c21', 'r06c22', 'r07c19', 'r07c21', 'r08c11', 'r08c22', 'r09c06', 'r09c16', 'r09c18', 'r10c12', 'r10c15', 'r11c10', 'r11c15', 'r11c17', 'r11c20', 'r12c13', 'r12c23', 'r14c07', 'r14c10', 'r14c13', 'r14c15', 'r14c17', 'r14c18', 'r14c19', 'r15c14', 'r15c22'])
    test_spheroid = '|'.join(['r02c04', 'r02c10', 'r02c11', 'r02c13', 'r02c14', 'r02c21', 'r02c23', 'r03c04', 'r03c07', 'r04c10', 'r04c14', 'r04c15', 'r04c17', 'r04c21', 'r04c23', 'r05c09', 'r05c11', 'r05c14', 'r05c15', 'r05c16', 'r05c17', 'r06c09', 'r06c13', 'r07c07', 'r07c16', 'r07c17', 'r07c18', 'r08c08', 'r08c12', 'r08c16', 'r09c10', 'r09c19', 'r09c20', 'r09c22', 'r10c13', 'r10c18', 'r10c20', 'r10c21', 'r10c23', 'r11c11', 'r11c13', 'r12c07', 'r12c08', 'r12c16', 'r12c21', 'r13c07', 'r13c15', 'r13c20', 'r13c21', 'r13c23', 'r14c09', 'r14c12', 'r14c20', 'r15c07', 'r15c08', 'r15c10', 'r15c15', 'r16c03', 'r16c22'])
    
    # 1 slice brightfield
    num_of_slices = 1
    
    df_train, df_val, df_test = create_train_and_val_df("/data/Nathalie_preprocessed_png/", num_of_slices, cutoff_first_slices, cutoff_last_slices, train_spheroid, val_spheroid, test_spheroid)

    ds_train = tf.data.Dataset.from_tensor_slices((df_train['nucleus'],
                                                    df_train['brightfield'], 
                                                    df_train['plane'],
                                                    df_train['number_of_slices']))

    ds_train = ds_train.shuffle((df_train.shape[0] // bs), reshuffle_each_iteration = True)
    ds_train = ds_train.map(read_imgs)

    ds_train = ds_train.batch(bs)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(buffer_size = tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((df_val['nucleus'],
                                                    df_val['brightfield'], 
                                                    df_val['plane'],
                                                    df_val['number_of_slices']))

    ds_val = ds_val.map(read_imgs)
    ds_val = ds_val.batch(1)
    ds_val = ds_val.prefetch(buffer_size = tf.data.AUTOTUNE)


    model = create_unet_model((1024, 1024, num_of_slices))
    opt = tf.keras.optimizers.Adam(learning_rate=1e-05)

    model.compile(optimizer = opt, loss = 'mse', metrics = ['mae', 'accuracy'])
    model.summary()

    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath = '/data/njanssen/models/nucleus/nucleus_020621_1plane/model.{epoch:02d}--{val_loss:.2f}.h5', monitor = 'val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir = '/data/njanssen/logs/nucleus/nucleus_020621_1plane')
    ]

    history = model.fit(ds_train, 
        epochs = 50, 
        steps_per_epoch = df_train.shape[0] // bs, 
        validation_data = ds_val, 
        validation_steps = df_val.shape[0] // bs, 
        callbacks = my_callbacks)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('nucleus/nucleus_020621_1plane.csv')

    model.save("nucleus/nucleus_020621_1plane.h5")


    # 3 plane brightfield
    num_of_slices = 3
    
    df_train, df_val, df_test = create_train_and_val_df("/data/Nathalie_preprocessed_png/", num_of_slices, cutoff_first_slices, cutoff_last_slices, train_spheroid, val_spheroid, test_spheroid)

    ds_train = tf.data.Dataset.from_tensor_slices((df_train['nucleus'],
                                                    df_train['brightfield'], 
                                                    df_train['plane'],
                                                    df_train['number_of_slices']))

    ds_train = ds_train.shuffle((df_train.shape[0] // bs), reshuffle_each_iteration = True)
    ds_train = ds_train.map(read_imgs)

    ds_train = ds_train.batch(bs)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(buffer_size = tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((df_val['nucleus'],
                                                    df_val['brightfield'], 
                                                    df_val['plane'],
                                                    df_val['number_of_slices']))

    ds_val = ds_val.map(read_imgs)
    ds_val = ds_val.batch(1)
    ds_val = ds_val.prefetch(buffer_size = tf.data.AUTOTUNE)


    model = create_unet_model((1024, 1024, num_of_slices))
    opt = tf.keras.optimizers.Adam(learning_rate=1e-05)

    model.compile(optimizer = opt, loss = 'mse', metrics = ['mae', 'accuracy'])
    model.summary()

    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath = '/data/njanssen/models/nucleus/nucleus_020621_3plane/model.{epoch:02d}--{val_loss:.2f}.h5', monitor = 'val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir = '/data/njanssen/logs/nucleus/nucleus_020621_3plane')
    ]

    history = model.fit(ds_train, 
        epochs = 50, 
        steps_per_epoch = df_train.shape[0] // bs, 
        validation_data = ds_val, 
        validation_steps = df_val.shape[0] // bs, 
        callbacks = my_callbacks)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('nucleus/nucleus_020621_3plane.csv')

    model.save("nucleus/nucleus_020621_3plane.h5")



    # 5 plane brightfield
    num_of_slices = 5

    df_train, df_val, df_test = create_train_and_val_df("/data/Nathalie_preprocessed_png/", num_of_slices, cutoff_first_slices, cutoff_last_slices, train_spheroid, val_spheroid, test_spheroid)

    ds_train = tf.data.Dataset.from_tensor_slices((df_train['nucleus'],
                                                    df_train['brightfield'], 
                                                    df_train['plane'],
                                                    df_train['number_of_slices']))

    ds_train = ds_train.shuffle((df_train.shape[0] // bs), reshuffle_each_iteration = True)
    ds_train = ds_train.map(read_imgs)

    ds_train = ds_train.batch(bs)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(buffer_size = tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((df_val['nucleus'],
                                                    df_val['brightfield'], 
                                                    df_val['plane'],
                                                    df_val['number_of_slices']))

    ds_val = ds_val.map(read_imgs)
    ds_val = ds_val.batch(1)
    ds_val = ds_val.prefetch(buffer_size = tf.data.AUTOTUNE)


    model = create_unet_model((1024, 1024, num_of_slices))
    opt = tf.keras.optimizers.Adam(learning_rate=1e-05)

    model.compile(optimizer = opt, loss = 'mse', metrics = ['mae', 'accuracy'])
    model.summary()

    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath = '/data/njanssen/models/nucleus/nucleus_020621_5plane/model.{epoch:02d}--{val_loss:.2f}.h5', monitor = 'val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir = '/data/njanssen/logs/nucleus/nucleus_020621_5plane')
    ]

    history = model.fit(ds_train, 
        epochs = 50, 
        steps_per_epoch = df_train.shape[0] // bs, 
        validation_data = ds_val, 
        validation_steps = df_val.shape[0] // bs, 
        callbacks = my_callbacks)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('nucleus/nucleus_020621_5plane.csv')

    model.save("nucleus/nucleus_020621_5plane.h5")


if __name__=="__main__":
    main()
