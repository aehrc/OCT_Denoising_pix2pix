# example of pix2pix gan for satellite to map image-to-image translation
#modified by Maryam Mehdizadeh - AEHRC CSIRO 2022-2023

import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import ZeroPadding2D

import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras.preprocessing.image import load_img, save_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint

import time
from tensorflow.python.keras.utils import conv_utils


vgg = VGG16(include_top=False,   input_shape=(512,512,3), weights = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' )
loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
loss_model.trainable = False


def perceptual_loss(y_true, y_pred):

	print('VGG MODEL IS CREATED')
	print(y_true.shape)
	print(y_pred.shape)
	y_true_c =   Concatenate()([y_true,y_true,y_true])    
	y_pred_c =   Concatenate()([y_pred,y_pred,y_pred])    


	return K.mean(K.square(loss_model(y_true_c) - loss_model(y_pred_c)))

def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true*y_pred)

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def res_block(input, filters, kernel_size=(3,3), strides=(1,1), use_dropout=False):
	"""
	Instanciate a Keras Resnet Block using sequential API.
	:param input: Input tensor
	:param filters: Number of filters to use
	:param kernel_size: Shape of the kernel for the convolution
	:param strides: Shape of the strides for the convolution
	:param use_dropout: Boolean value to determine the use of dropout
	:return: Keras Model
	"""
	x = ZeroPadding2D((1,1))(input)
	x = Conv2D(filters=filters,
			   kernel_size=kernel_size,
			   strides=strides,)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	if use_dropout:
		x = Dropout(0.5)(x)

	x = ZeroPadding2D((1,1))(x)
	x = Conv2D(filters=filters,
				kernel_size=kernel_size,
				strides=strides,)(x)
	x = BatchNormalization()(x)

	# Two convolution layers followed by a direct connection between input and output
	merged = Add()([input, x])
	
	return merged

ngf = 64
input_nc = 1
output_nc = 1
input_shape_generator = (448, 896, input_nc)
n_blocks_gen = 16


def define_generator(image_shape):
	"""Build generator architecture."""
	# Current version : ResNet block
	inputs = Input(shape=image_shape)

	x = ZeroPadding2D((3, 3))(inputs)
	x = Conv2D(filters=ngf, kernel_size=(7,7), padding='valid')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# Increase filter number
	n_downsampling = 2
	for i in range(n_downsampling):
		mult = 2**i
		x = Conv2D(filters=ngf*mult*2, kernel_size=(3,3), strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

	# Apply 9 ResNet blocks
	mult = 2**n_downsampling
	for i in range(n_blocks_gen):
		x = res_block(x, ngf*mult, use_dropout=True)

	# Decrease filter number to 3 (RGB)
	for i in range(n_downsampling):
		mult = 2**(n_downsampling - i)
		x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3,3), strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

	x = ZeroPadding2D((3,3))(x)
	x = Conv2D(filters=output_nc, kernel_size=(7,7), padding='valid')(x)
	x = Activation('tanh')(x)

	# Add direct connection from input to output and recenter to [-1, 1]
	outputs = Add()([x, inputs])
	outputs = Lambda(lambda z: z/2)(outputs)

	model = Model(inputs=inputs, outputs=outputs, name='Generator')
	return model

ndf = 64
output_nc = 1
input_shape_discriminator = (512, 512, output_nc)


def define_discriminator(image_shape):
	"""Build discriminator architecture."""
	n_layers, use_sigmoid = 3, False
	inputs = Input(shape=image_shape)

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

def define_gan(g_model, d_model, image_shape):
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False

	inputs = Input(shape=image_shape)
	generated_images = g_model(inputs)
	outputs = d_model(generated_images)
	model = Model(inputs=inputs, outputs=[outputs, generated_images])
	opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss=[wasserstein_loss, perceptual_loss], optimizer=opt, loss_weights=[1,100])
	return model

#  load and prepare training images
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

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	print("samples size is:")
	print(samples.shape)
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = -ones((len(X), patch_shape, patch_shape, 1))
	return X, y

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
		save_img("./test_images_resnet/test_" + str(i) + "_" +  str(step)+ ".png", tf.concat([X_realAi, X_fakeBi, X_realBi],1))
		
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
	for i in range(56210,n_steps):
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

dataset = load_real_samples('./averaged_noisy_512_test.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# print(image_shape)
# define the models
d_model = define_discriminator(image_shape)


print("defining generator model")
g_model = define_generator(image_shape)

g_model = load_model('./saved_models_resnet/model_073000.h5', compile=False)

# define the composite model
print("defining GAN model")
gan_model = define_gan(g_model, d_model, image_shape)
# train model
# print("start to train model")
# train(d_model, g_model, gan_model, dataset)

print("start to test model")
test(73000,g_model, dataset)
