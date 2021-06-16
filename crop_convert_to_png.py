import os
import pandas as pd
import skimage.io

coords = pd.read_csv('Nathalie_786-O_DSRT_crop_coords.csv', sep = ';')
img_dir = 'Nathalie_786-O_DSRT_H1FS2A/'
target_dir = 'Nathalie_preprocessed_png/'

count = 1
target_size = 1024

for filename in os.listdir(img_dir):
    img_name = filename.split('.')[0]
    for spheroid in list(coords.spheroid):
        if filename.rsplit('f01',1)[0].startswith(spheroid):
            coordinates = coords[coords['spheroid'].str.contains(spheroid)]
            x, y = int(coordinates['x']), int(coordinates['y'])
            img = skimage.io.imread(img_dir + filename)
            crop_img = img[y:y+target_size, x:x+target_size]
            skimage.io.imsave(target_dir+img_name+'.png', crop_img)
            print('Images cropped: ', count)
            count += 1
