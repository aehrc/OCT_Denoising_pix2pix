# example of pix2pix gan for satellite to map image-to-image translation
#modified by Maryam Mehdizadeh - AEHRC CSIRO 2022-2023

import tensorflow.compat.v1 as tf


from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Subtract
from matplotlib import pyplot
from tensorflow.python.keras.preprocessing.image import load_img, save_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# import tensorflow as tf
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from  tensorflow.keras.layers import Concatenate

from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import UpSampling2D

import time


vgg = VGG16(include_top=False,   input_shape=(512,512,3), weights = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' )
loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
loss_model.trainable = False


def perceptual_loss(y_true, y_pred):

	print(y_true.shape)
	print(y_pred.shape)
	y_true_c =   Concatenate()([y_true,y_true,y_true])    
	y_pred_c =   Concatenate()([y_pred,y_pred,y_pred])    


	return K.mean(K.square(loss_model(y_true_c) - loss_model(y_pred_c)))

def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true*y_pred)


ndf = 64
output_nc = 1
input_shape_discriminator = (512, 512, output_nc)
def define_discriminator(image_shape):
	"""Build discriminator architecture."""
	n_layers, use_sigmoid = 3, False
	inputs = Input(shape=input_shape_discriminator)

	x = Conv2D(filters=ndf, kernel_size=(4,4), strides=2, padding='same')(inputs)
	x = LeakyReLU(0.2)(x)

	nf_mult = 1 
	for n in range(n_layers):
		nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
		x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(0.2)(x)

	nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
	x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=1, padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(0.2)(x)

	x = Conv2D(filters=1, kernel_size=(4,4), strides=1, padding='same')(x)
	if use_sigmoid:
		x = Activation('sigmoid')(x)

	x = Flatten()(x)
	x = Dense(1024, activation='tanh')(x)
	x = Dense(1, activation='sigmoid')(x)
	opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	
	model = Model(inputs=inputs, outputs=x, name='Discriminator')
	model.compile(optimizer=opt, loss=wasserstein_loss)
	return model

image_shape = (512,512,1)
#UNET

def res_block(x, nb_filters, strides):
	res_path = BatchNormalization()(x)
	res_path = Activation(activation='relu')(res_path)
	res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
	res_path = BatchNormalization()(res_path)
	res_path = Activation(activation='relu')(res_path)
	res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

	shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
	shortcut = BatchNormalization()(shortcut)

	res_path = Add()([shortcut, res_path])
	return res_path


def encoder(x):
	to_decoder = []

	main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
	main_path = BatchNormalization()(main_path)
	main_path = Activation(activation='relu')(main_path)

	main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

	shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
	shortcut = BatchNormalization()(shortcut)

	main_path = Add()([shortcut, main_path])
	# first branching to decoder
	to_decoder.append(main_path)

	main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
	to_decoder.append(main_path)

	main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
	to_decoder.append(main_path)

	return to_decoder


def decoder(x, from_encoder):
	main_path = UpSampling2D(size=(2, 2))(x)
	main_path = Concatenate()([main_path, from_encoder[2]])
	main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

	main_path = UpSampling2D(size=(2, 2))(main_path)
	main_path = Concatenate()([main_path, from_encoder[1]])
	main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

	main_path = UpSampling2D(size=(2, 2))(main_path)
	main_path = Concatenate()([main_path, from_encoder[0]])
	main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

	return main_path


def define_generator(image_shape):
	inputs = Input(shape=image_shape)

	to_decoder = encoder(inputs)

	path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

	path = decoder(path, from_encoder=to_decoder)

	path = Conv2D(filters=1, kernel_size=(1, 1), activation='tanh')(path)

	return Model(inputs=inputs, outputs=path)


def define_gan(g_model, d_model, image_shape):
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False


	inputs = Input(shape=image_shape)
	generated_images = g_model(inputs)
	outputs = d_model(generated_images)
	model = Model(inputs=inputs, outputs=[outputs, generated_images])
	opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss=[wasserstein_loss, "mae"], optimizer=opt, loss_weights=[1,100])
	return model

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [0,1]
	X1 = (X1) / 255.0
	X2 = (X2) / 255.0
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# select a batch of random samples, returns images and target
def generate_test_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	# ix = randint(0, trainA.shape[0], n_samples)
	ix = list(range(0,trainA.shape[0]))
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	print("samples size is:")
	print(samples.shape)
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = -ones((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=20):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [0,1] to [0,255.0]

	# X_realA = (X_realA + 1) / 2.0
	# X_realB = (X_realB + 1) / 2.0
	# X_fakeB = (X_fakeB + 1) / 2.0

	PSNR = []
	PSNR_original = []
	# plot real source images
	for i in range(n_samples):
	
		X_realAi = img_to_array(X_realA[i])

		X_realBi = img_to_array(X_realB[i])
		X_fakeBi = img_to_array(X_fakeB[i])
		X_realAi = np.clip(255 * X_realAi, 0, 255).astype('uint8')
		X_realBi = np.clip(255 * X_realBi, 0, 255).astype('uint8')
		X_fakeBi = np.clip(255 * X_fakeBi, 0, 255).astype('uint8')
		save_img("./evaluate_images/noisy_" + str(i) + "_" +  str(step+1)+ ".png", tf.concat([X_realAi, X_fakeBi, X_realBi],1))
		# save_img("./evaluate_images/noisy_" + str(i) + "_" +  str(step+1)+ ".png", tf.concat([X_realAi, X_fakeBi, X_realBi],1))
	
		psnr = cal_psnr(X_realB[i], X_fakeB[i])
		PSNR.append(psnr)
		print(i,psnr)
		if(i < 1):
			psnr_noisy = cal_psnr(X_realBi, X_realAi)
			PSNR_original.append(psnr_noisy)

	print(np.mean(PSNR))
	print(np.mean(PSNR_original))
	filename2 = './saved_models/model_%06d.h5' % (step+1)
	# filename2 = './saved_models//model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s model' % (filename2))


def summarize_performance_test(step, g_model, dataset, n_samples=200):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_test_samples(dataset, n_samples, 1)
	# generate a batch of fake samples


	print("generating fake samples")
	start = time.time()
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	finish = time.time() - start

	print("generating fake samples took: " + str(finish) + " seconds")

	PSNR = []
	PSNR_original = []
	# plot real source images
	for i in range(n_samples):
		X_realAi = img_to_array(X_realA[i])
		X_realBi = img_to_array(X_realB[i])
		X_fakeBi = img_to_array(X_fakeB[i])
		save_img("./test_" + str(i) + "_" +  str(step)+ ".png", tf.concat([X_realAi, X_fakeBi, X_realBi],1))
		
		psnr = cal_psnr(X_realB[i], X_fakeB[i])
		PSNR.append(psnr)
		# print(i,psnr)
		if(i < 1):
			psnr_noisy = cal_psnr(X_realBi, X_realAi)
			PSNR_original.append(psnr_noisy)
	
	print(np.mean(PSNR))
	print(np.mean(PSNR_original))
	# filename2 = '/content/drive/MyDrive/pix2pix/model_%06d.h5' % (step+1)
	# g_model.save(filename2)
	# print('>Saved: %s model' % (filename2))


def tf_psnr(im1, im2):
	# assert pixel value range is 0-1
	mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
	return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))

def cal_psnr(im1, im2):
	# assert pixel value range is 0-255 and type is uint8
	mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
	psnr = 10 * np.log10(255 ** 2 / mse)
	return psnr
 
# train pix2pix models
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=200, n_batch=4):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(49640,n_steps):
		start = time.time()
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# print(X_realA.shape, X_realB.shape, y_real.shape)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		finish = time.time() - start
		print("each apoch takes: " + str(finish) + " seconds")
		# print('>%d, d1[%.3f] d2[%.3f]' % (i+1, d_loss1, d_loss2))
		# summarize model performance
		if (i+1) % (730) == 0:
			summarize_performance(i, g_model, dataset)
# def evaluate(g_model, dataset):
# 	summarize_performance(step, g_model, dataset)
 
def test(step,g_model, dataset):
	summarize_performance_test(step,g_model, dataset)



# load image data
print("Loading data")
dataset = load_real_samples('./averaged_noisy_512_test.npz')
# finish = time.time() - start

# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# print(image_shape)
# define the models
d_model = define_discriminator(image_shape)

print("defining generator model")
g_model = define_generator(image_shape)



g_model = load_model('./saved_models_unet_wgan_mae/model_073000.h5', compile=False)
# finish = time.time() - start

# define the composite model
print("defining GAN model")
gan_model = define_gan(g_model, d_model, image_shape)
# train model
print("start to test model")
# train(d_model, g_model, gan_model, dataset)
test(73000,g_model, dataset)
