import tensorflow as tf
import os
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
#                          Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
# from keras.optimizers import Adam, RMSprop
# from keras.layers.wrappers import Bidirectional
# from keras.layers.merge import add
# from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
# from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

BASE_DIR = os.path.dirname(__file__)
#UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/images')
MODEL_DIR = os.path.join(BASE_DIR, 'model/model_30.h5')
MODEL_DIR_encode = os.path.join(BASE_DIR, 'model/encode.h5')

max_length= 34
vocab_size= 1652
embedding_dim=200


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    print("naruto naruto")
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

vocabs = list()
file = open("vocab.txt", 'r')
text = file.read()
file.close()
for word in text.split('\n'):
  vocabs.append(word)
ixtoword = {}
wordtoix = {}

ix = 1
for w in vocabs:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

def encode(image):
  im= preprocess(image)
  fea_vec= encoder.predict(im)
  fea_vec= np.reshape(fea_vec,fea_vec.shape[1])
  return fea_vec


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


#load model 
# global model,encoder
def load_keras_model():
	from keras.models import model_from_json
	global model,encoder
	json_file = open('./model/model_im.json','r')
	load_model_json= json_file.read()
	json_file.close()
	model = model_from_json(load_model_json)
	model.load_weights("./model/model_30.h5")
	encoder = load_model("./model/encode.h5")
	# return model, encoder
	# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	# x_train /= 255
	# x_test /= 255
	# model.evaluate(x_test, y_test)

# def load_images_to_data(image_directory):
#     # list_of_files = os.listdir(image_directory)
#     # for file in list_of_files:
#     #     image_file_name = os.path.join(image_directory, file)
#     #     if ".png" in image_file_name:
#     img = Image.open(image_directory).convert("L")
#     img = np.resize(img, (28,28,1))
#     im2arr = np.array(img)
#     im2arr = im2arr.reshape(1,28,28,1)
#           #  features_data = np.append(features_data, im2arr, axis=0)
#            # label_data = np.append(label_data, [image_label], axis=0)
#     return im2arr

def hienthi_kq(image_directory):
	img = encode(image_directory)
	im = img.reshape((1,2048))
	#cv2.imwrite('/home/duc_mnsd/Desktop/download.png',x_test[image_index])
	# plt.imshow(img.reshape(28, 28),cmap='Greys')
	# pred = model.predict(img.reshape(1, 28,28, 1))
	# kq = pred.argmax()
	return greedySearch(im)
