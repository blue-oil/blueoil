from six.moves import cPickle as pickle
from  PIL import Image
import numpy as np
import scipy.misc

f = open('dataset/CIFAR_10/cifar-10-batches-py/data_batch_1', 'rb')

tupled_data= pickle.load(f, encoding='bytes')

f.close()

img = tupled_data[b'data']
label = tupled_data[b'labels']

for i in range(10):
    single_img = np.array(img[i])
    single_label = str(label[i])
    single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
    scipy.misc.imsave(f'images/{i}-{single_label}.jpg',single_img_reshaped)
    #single_img_reshaped = single_img.reshape(32,32,3)

#plt.imshow(single_img_reshaped)
