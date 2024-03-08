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

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(512,512,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model


def define_gan(g_model, d_model, image_shape):
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	inputs = Input(shape=image_shape)
	generated_images = g_model(inputs)
	outputs = d_model(generated_images)
	model = Model(inputs=inputs, outputs=[outputs, generated_images])
	opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	# opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=[wasserstein_loss, perceptual_loss], optimizer=opt, loss_weights=[1,100])
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
		save_img("./test_images_unet54/test_" + str(i) + "_" +  str(step)+ ".png", tf.concat([X_realAi, X_fakeBi, X_realBi],1))
		
		psnr = cal_psnr(X_realB[i], X_fakeB[i])
		PSNR.append(psnr)
		# print(i,psnr)
		if(i < 1):
			psnr_noisy = cal_psnr(X_realBi, X_realAi)
			PSNR_original.append(psnr_noisy)
	
	print(np.mean(PSNR))
	print(np.mean(PSNR_original))



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
	for i in range(45990,n_steps):
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

# start = time.time()
dataset = load_real_samples('./averaged_noisy_512_test.npz')
# finish = time.time() - start

# print("Loading data took: " + str(finish) + " seconds")
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# print(image_shape)
# define the models
d_model = define_discriminator(image_shape)

print("defining generator model")
g_model = define_generator(image_shape)

# start = time.time()
g_model = load_model('./model_073000.h5', compile=False)
# finish = time.time() - start


# define the composite model
print("defining GAN model")
gan_model = define_gan(g_model, d_model, image_shape)
# train model

# print("start to train the model")
# train(d_model, g_model, gan_model, dataset)

print("start to test the model")
test(73000,g_model, dataset)
