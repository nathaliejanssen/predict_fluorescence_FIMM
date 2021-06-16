# STEP 3 CREATE CSV WITH MSE PER PREDICTION
import os
import numpy as np
from pathlib import Path
import pandas as pd
import imageio

def create_df(path_to_gt, num_of_slices, cutoff_first_slices, cutoff_last_slices, tested_spheroids, path_to_pred):
    img_path = Path(path_to_gt)
    nucleus = [x.as_posix() for x in img_path.glob('*ch1*.png')]
    nucleus.sort()
    df_imgs = pd.DataFrame(data={'nucleus': nucleus})

    df_imgs['plane'] = df_imgs['nucleus'].apply(lambda x: int(x.split('-')[0].split('p')[-1]))
    df_imgs = df_imgs[(df_imgs.plane >= cutoff_first_slices)]
    df_imgs = df_imgs[(df_imgs.plane <= cutoff_last_slices)]
    df_imgs = df_imgs[(df_imgs.plane <= (df_imgs.plane.max() - (num_of_slices//2))) & (df_imgs.plane >= (df_imgs.plane.min() + (num_of_slices//2)))]
    
    df_imgs['number_of_slices'] = num_of_slices
    df_imgs = df_imgs.astype({col: 'int32' for col in df_imgs.select_dtypes('int64').columns})
    df_imgs['spheroid'] = df_imgs['nucleus'].apply(lambda x: x.split('/')[-1].split('-')[0])
    
    df_imgs = df_imgs[df_imgs['nucleus'].str.contains(tested_spheroids)]
    
    predictions = []
    for i in df_imgs['spheroid']:
        for filename in os.listdir(Path(path_to_pred)):
            if i in filename:
                predictions.append(path_to_pred + filename)
    df_imgs['predicted'] = predictions

    return df_imgs

def mse(gt, pred):
    gt = imageio.imread(gt).astype("float") / 65535
    pred = imageio.imread(pred).astype("float") / 65535

    err = np.sum((gt.astype("float") - pred.astype("float")) ** 2)
    err /= float(gt.shape[0] * pred.shape[1])

    return '{:.10f}'.format(err)


# CONFIG
path_to_gt = '/data/Nathalie_preprocessed_png/'
cutoff_first_slices = 3
cutoff_last_slices = 10

# 1-plane
path_to_pred = 'nucleus/pred_020621/1-plane/'
num_of_slices = 1

tested_spheroids = []
for filename in os.listdir(path_to_pred):
    tested_spheroids.append(filename.split('-')[0])
tested_spheroids = '|'.join(tested_spheroids)

# Actual computation
df_img = create_df(path_to_gt, num_of_slices, cutoff_first_slices, cutoff_last_slices, tested_spheroids, path_to_pred)

df_img['mse'] = df_img.apply(lambda row: mse(row['nucleus'], row['predicted']), axis=1)
df_img.to_csv('nucleus_020621_1plane.csv')



# 3-plane
path_to_pred = 'nucleus/pred_020621/3-plane/'
num_of_slices = 3

tested_spheroids = []
for filename in os.listdir(path_to_pred):
    tested_spheroids.append(filename.split('-')[0])
tested_spheroids = '|'.join(tested_spheroids)

# Actual computation
df_img = create_df(path_to_gt, num_of_slices, cutoff_first_slices, cutoff_last_slices, tested_spheroids, path_to_pred)

df_img['mse'] = df_img.apply(lambda row: mse(row['nucleus'], row['predicted']), axis=1)
df_img.to_csv('nucleus_020621_3plane.csv')



# 5-plane
path_to_pred = 'nucleus/pred_020621/5-plane/'
num_of_slices = 5

tested_spheroids = []
for filename in os.listdir(path_to_pred):
    tested_spheroids.append(filename.split('-')[0])
tested_spheroids = '|'.join(tested_spheroids)

# Actual computation
df_img = create_df(path_to_gt, num_of_slices, cutoff_first_slices, cutoff_last_slices, tested_spheroids, path_to_pred)

df_img['mse'] = df_img.apply(lambda row: mse(row['nucleus'], row['predicted']), axis=1)
df_img.to_csv('nucleus_020621_5plane.csv')
