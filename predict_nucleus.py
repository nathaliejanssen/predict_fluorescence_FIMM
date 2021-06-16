# STEP 2 PREDICT ALL NUCLEUS IMGS

import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf
import random
import imageio
import matplotlib.pyplot as plt


def create_img_path_df(path_to_imgfolder, num_of_slices, cutoff_first_slices, cutoff_last_slices, test_spheroid):
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


    df_test = df_imgs[df_imgs['nucleus'].str.contains(test_spheroid)]

    return df_test

def read_imgs_test(nucleus, brightfield, plane, number_of_slices):
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

    return bfimg

def main():
    # CONFIG
    bs = 1
    cutoff_first_slices = 3
    cutoff_last_slices = 10
    test_spheroid = '|'.join(['r02c04', 'r02c10', 'r02c11', 'r02c13', 'r02c14', 'r02c21', 'r02c23', 'r03c04', 'r03c07', 'r04c10', 'r04c14', 'r04c15', 'r04c17', 'r04c21', 'r04c23', 'r05c09', 'r05c11', 'r05c14', 'r05c15', 'r05c16', 'r05c17', 'r06c09', 'r06c13', 'r07c07', 'r07c16', 'r07c17', 'r07c18', 'r08c08', 'r08c12', 'r08c16', 'r09c10', 'r09c19', 'r09c20', 'r09c22', 'r10c13', 'r10c18', 'r10c20', 'r10c21', 'r10c23', 'r11c11', 'r11c13', 'r12c07', 'r12c08', 'r12c16', 'r12c21', 'r13c07', 'r13c15', 'r13c20', 'r13c21', 'r13c23', 'r14c09', 'r14c12', 'r14c20', 'r15c07', 'r15c08', 'r15c10', 'r15c15', 'r16c03', 'r16c22'])


    # PREDICT 1 SLICE
    num_of_slices = 1
    df_test = create_img_path_df("/data/Nathalie_preprocessed_png/", num_of_slices, cutoff_first_slices, cutoff_last_slices, test_spheroid)

    ds_test = tf.data.Dataset.from_tensor_slices((df_test['nucleus'],
                                                    df_test['brightfield'], 
                                                    df_test['plane'],
                                                    df_test['number_of_slices']))

    ds_test = ds_test.map(read_imgs_test)
    ds_test = ds_test.batch(batch_size = bs, drop_remainder = True)

    model = tf.keras.models.load_model("nucleus/nucleus_020621_1plane.h5")

    predictions = model.predict(ds_test)

    names = []
    for filename in df_test['nucleus']:
        filename = filename.split('/')[-1].split('.')[0]
        names.append(filename+'_1plane_pred.png')

    a = 0
    for pred in predictions:
        y_pred = pred * 65535
        y_pred = np.squeeze(y_pred, axis = 2)
        y_pred = y_pred.astype(np.uint16)
        imageio.imwrite('nucleus/pred_020621/1-plane/'+names[a], y_pred)
        print(a)
        a += 1
    print('1plane done')


    # PREDICT 3 SLICES
    num_of_slices = 3
    df_test = create_img_path_df("/data/Nathalie_preprocessed_png/", num_of_slices, cutoff_first_slices, cutoff_last_slices, test_spheroid)

    ds_test = tf.data.Dataset.from_tensor_slices((df_test['nucleus'],
                                                    df_test['brightfield'], 
                                                    df_test['plane'],
                                                    df_test['number_of_slices']))

    ds_test = ds_test.map(read_imgs_test)
    ds_test = ds_test.batch(batch_size = bs, drop_remainder = True)

    model = tf.keras.models.load_model("nucleus/nucleus_020621_3plane.h5")

    predictions = model.predict(ds_test)

    names = []
    for filename in df_test['nucleus']:
        filename = filename.split('/')[-1].split('.')[0]
        names.append(filename+'_3plane_pred.png')

    a = 0
    for pred in predictions:
        y_pred = pred * 65535
        y_pred = np.squeeze(y_pred, axis = 2)
        y_pred = y_pred.astype(np.uint16)
        imageio.imwrite('nucleus/pred_020621/3-plane/'+names[a], y_pred)
        print(a)
        a += 1
    print('3plane done')


    # PREDICT 5 SLICES
    num_of_slices = 5
    df_test = create_img_path_df("/data/Nathalie_preprocessed_png/", num_of_slices, cutoff_first_slices, cutoff_last_slices, test_spheroid)

    ds_test = tf.data.Dataset.from_tensor_slices((df_test['nucleus'],
                                                    df_test['brightfield'], 
                                                    df_test['plane'],
                                                    df_test['number_of_slices']))

    ds_test = ds_test.map(read_imgs_test)
    ds_test = ds_test.batch(batch_size = bs, drop_remainder = True)

    model = tf.keras.models.load_model("nucleus/nucleus_020621_5plane.h5")

    predictions = model.predict(ds_test)

    names = []
    for filename in df_test['nucleus']:
        filename = filename.split('/')[-1].split('.')[0]
        names.append(filename+'_5plane_pred.png')

    a = 0
    for pred in predictions:
        y_pred = pred * 65535
        y_pred = np.squeeze(y_pred, axis = 2)
        y_pred = y_pred.astype(np.uint16)
        imageio.imwrite('nucleus/pred_020621/5-plane/'+names[a], y_pred)
        print(a)
        a += 1
    print('5plane done')
    print('All Done')
  
    

if __name__=="__main__":
    main()
