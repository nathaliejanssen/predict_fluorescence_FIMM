# STEP 1 TRAIN
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

from config import config

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate


def create_train_and_val_df(path_to_imgfolder, num_of_slices, cutoff_first_slices, cutoff_last_slices, train_spheroid, val_spheroid):
    img_path = Path(path_to_imgfolder)
    nucleus = [x.as_posix() for x in img_path.glob('*ch1*.png')]
    nucleus.sort()
    
    draq7 = [x.as_posix() for x in img_path.glob('*ch2*.png')]
    draq7.sort()
    
    cellmask = [x.as_posix() for x in img_path.glob('*ch3*.png')]
    cellmask.sort()

    brightfield = [x.as_posix() for x in img_path.glob('*ch4*.png')]
    brightfield.sort()
    
    mitotracker = [x.as_posix() for x in img_path.glob('*ch5*.png')]
    mitotracker.sort()

    df_imgs = pd.DataFrame(data={'nucleus': nucleus, 
                                 'draq7': draq7, 
                                 'cellmask': cellmask,
                                 'mitotracker': mitotracker, 
                                 'brightfield': brightfield})

    df_imgs['plane'] = df_imgs['brightfield'].apply(lambda x: int(x.split('-')[0].split('p')[-1]))
    df_imgs = df_imgs[(df_imgs.plane >= cutoff_first_slices)]
    df_imgs = df_imgs[(df_imgs.plane <= cutoff_last_slices)]
    df_imgs = df_imgs[(df_imgs.plane <= (df_imgs.plane.max() - (num_of_slices//2))) & (df_imgs.plane >= (df_imgs.plane.min() + (num_of_slices//2)))]
    
    df_imgs['number_of_slices'] = num_of_slices

    df_imgs = df_imgs.astype({col: 'int32' for col in df_imgs.select_dtypes('int64').columns})

    df_train = df_imgs[df_imgs['nucleus'].str.contains(train_spheroid)]
    df_val = df_imgs[df_imgs['nucleus'].str.contains(val_spheroid)]

    return df_train, df_val

def decode_imgs(channel):
    img = tf.io.decode_png(tf.io.read_file(channel), channels = 1, dtype = tf.uint16) 
    img = tf.cast(img, dtype = tf.float32)
    img = tf.math.log(img)
    img = tf.math.divide(img, 65535)
    img = tf.image.resize_with_crop_or_pad(img, config.img_size, config.img_size)

    return img

def read_imgs(nucleus, draq7, cellmask, mitotracker, brightfield, plane, number_of_slices):
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
        
        img = decode_imgs(fn)
        imgs = imgs.write(i, tf.squeeze(img))
    bfimg = imgs.stack()    
    bfimg = tf.transpose(bfimg, perm = [1,2,0])    
    
    if config.all_channels == True:
        fluo_target = tf.concat([decode_imgs(nucleus), decode_imgs(draq7), decode_imgs(cellmask), decode_imgs(mitotracker)], axis=2)
    
    else:
        if config.channel == 'nucleus':
            fluo_target = decode_imgs(nucleus)
        elif config.channel == 'draq7':
            fluo_target = decode_imgs(draq7)
        elif config.channel == 'cellmask':
            fluo_target = decode_imgs(cellmask)
        elif config.channel == 'mitotracker':
            fluo_target = decode_imgs(mitotracker)

    return bfimg, fluo_target

def create_unet_model(input_shape, num_fluo_channels):
    def down(connected_layer, num_filters, pool = True):
        conv = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer = 'he_uniform')(connected_layer)
        conv = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer = 'he_uniform')(conv)

        if pool == True:
            pool = MaxPool2D(2)(conv)
            return conv, pool
        else:
            return conv

    def up(connected_layer, concat_layer, num_filters):
        up = UpSampling2D(2)(connected_layer)
        up = Conv2D(num_filters, 2, padding="same", activation = 'relu', kernel_initializer='he_uniform')(up)
        up = Concatenate(axis=3)([concat_layer, up])

        conv = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(up)
        conv = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(conv)
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
    out = Conv2D(num_fluo_channels, 1, activation=None)(c9)

    model = Model(inputs=unet_input, outputs = out, name = 'U-net')
    
    return model

def custom_mse(y_true, y_pred):
    weights = np.array([0.1, 1, 0.1, 0.1])
    calc = K.square(y_pred - y_true) * weights
    
    return K.mean(calc, axis=-1)

def main():
    df_train, df_val = create_train_and_val_df(config.input_dir, 
                                               config.cutoff_first_slices, 
                                               config.cutoff_last_slices, 
                                               config.train_spheroid, 
                                               config.val_spheroid)

    ds_train = tf.data.Dataset.from_tensor_slices((df_train['nucleus'],
                                                   df_train['draq7'],
                                                   df_train['mitotracker'],
                                                   df_train['cellmask'],
                                                   df_train['brightfield'], 
                                                   df_train['plane'],
                                                   df_train['number_of_slices']))
    ds_train = ds_train.shuffle((df_train.shape[0] // config.bs), reshuffle_each_iteration = True)
    ds_train = ds_train.map(read_imgs)
    
    ds_train = ds_train.batch(config.bs).repeat()
    ds_train = ds_train.prefetch(buffer_size = tf.data.AUTOTUNE)
    
    ds_val = tf.data.Dataset.from_tensor_slices((df_val['nucleus'],
                                                 df_val['draq7'],
                                                 df_val['mitotracker'],
                                                 df_val['cellmask'],
                                                 df_val['brightfield'], 
                                                 df_val['plane'],
                                                 df_val['number_of_slices']))    
    ds_val = ds_val.map(read_imgs)
    ds_val = ds_val.batch(1)
    ds_val = ds_val.prefetch(buffer_size = tf.data.AUTOTUNE)

    opt = tf.keras.optimizers.Adam(learning_rate = config.lr)
    
    
    if config.all_channels == True:
        model = create_unet_model(input_shape = (config.img_size, config.img_size, config.num_of_slices), num_fluo_channels = 4)
        model.compile(optimizer = opt, loss = custom_mse, metrics = config.metrics)
    else:        
        model = create_unet_model(input_shape = (config.img_size, config.img_size, config.num_of_slices), num_fluo_channels = 1)
        model.compile(optimizer = opt, loss = config.loss, metrics = config.metrics)
    
    
    model.summary()

    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath = config.model_dir, monitor = 'val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir = config.log_dir)
    ]

    history = model.fit(ds_train, 
                        epochs = config.epochs, 
                        steps_per_epoch = df_train.shape[0] // config.bs, 
                        validation_data = ds_val, 
                        validation_steps = df_val.shape[0] // config.bs, 
                        callbacks = my_callbacks)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(config.history_filename)

    model.save(config.model_name)




if __name__=="__main__":
    main()