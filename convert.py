import os
import sys
import math
import time

import numpy as np

from tlc.net import NeuralNetwork
from tlc.bundle import *
from caffe.caffe_pb2 import NetParameter, LayerParameter, PoolingParameter

FunctionMapping = {
	'rlinear': LayerParameter.RELU,
	'sigmoid': LayerParameter.SIGMOID,
	'softmax': LayerParameter.SOFTMAX,
}

def convert(tlcNet):
	caffeNet = NetParameter()
	inputLayer = tlcNet.layers[0]
	caffeNet.input.append(inputLayer.name)
	# print caffeNet.input
	caffeNet.input_dim.append(1) # no crops for input image
	if type(inputLayer.dimOutput) is list:
		caffeNet.input_dim.extend(inputLayer.dimOutput)
	else:
		bundle = tlcNet.layers[1].bundle
		if type(bundle) is FullBundle:
			caffeNet.input_dim.append(1)
			caffeNet.input_dim.append(1)
			caffeNet.input_dim.append(inputLayer.dimOutput)
		else:
			caffeNet.input_dim.extend(bundle.geo.dimInput)

	# copy basic info from tlc to caffe
	for i in range(1, len(tlcNet.layers)):
		tlcLayer = tlcNet.layers[i]
		layers = convertLayer(tlcLayer)
		caffeNet.layers.extend(layers)

	caffeNet.layers[-1].top[0] = "prob"
	# update bottom and top info

	return caffeNet
	
def convertLayer(tlcLayer):
	ret = []
	bundle = tlcLayer.bundle
	# create caffe layer
	caffeLayer = LayerParameter()
	ret.append(caffeLayer)

	caffeLayer.name = tlcLayer.name
	caffeLayer.bottom.append(bundle.input.name)
	caffeLayer.top.append(tlcLayer.name)

	if type(bundle) is FullBundle:
		caffeLayer.type = LayerParameter.INNER_PRODUCT
		caffeLayer.inner_product_param.num_output = tlcLayer.dimOutput
		# add computation layer
		computLayer = LayerParameter()
		ret.append(computLayer)
		computLayer.name = "%s.comput" % tlcLayer.name
		computLayer.bottom.append(tlcLayer.name)
		computLayer.top.append(tlcLayer.name)
		computLayer.type = FunctionMapping[tlcLayer.outputFunc]
	elif type(bundle) is ConvolveBundle:
		if bundle.sharing is not None and not bundle.sharing[-1]:
			# local convolution
			caffeLayer.type = LayerParameter.LOCAL
			conv = caffeLayer.local_param
		else:
			# global convolution
			caffeLayer.type = LayerParameter.CONVOLUTION
			conv = caffeLayer.convolution_param

		conv.kernel_size = bundle.geo.dimKernel[-1]
		if bundle.geo.stride is not None:
			conv.stride = bundle.geo.stride[-1]
		else:
			conv.stride = 1
		if type(bundle.mapCount) is list:
			conv.num_output = bundle.mapCount[0]
		else:
			conv.num_output = bundle.mapCount
		if bundle.geo.padding is not None and bundle.geo.padding[-1]:
			conv.pad = (conv.kernel_size - 1) / 2
		# add computation layer
		computLayer = LayerParameter()
		ret.append(computLayer)
		computLayer.name = "%s.comput" % tlcLayer.name
		computLayer.bottom.append(tlcLayer.name)
		computLayer.top.append(tlcLayer.name)
		computLayer.type = FunctionMapping[tlcLayer.outputFunc]

	elif type(bundle) is ResponseNormBundle:
		caffeLayer.type = LayerParameter.LRN
		lrn_param = caffeLayer.lrn_param
		lrn_param.local_size = bundle.geo.dimKernel[0]
		lrn_param.alpha = bundle.alpha
		lrn_param.beta = bundle.beta
		lrn_param.k = bundle.offset 
	elif type(bundle) is MaxPoolBundle:
		caffeLayer.type = LayerParameter.POOLING
		pooling_param = caffeLayer.pooling_param
		pooling_param.pool = PoolingParameter.MAX
		pooling_param.kernel_size = bundle.geo.dimKernel[-1]
		pooling_param.stride = bundle.geo.stride[-1]
		if bundle.geo.padding is not None and bundle.geo.padding[-1]:
			pooling_param.pad = (pooling_param.kernel_size - 1) / 2

	return ret

def convertBlobs(tlcNet, caffeNet):
	# copy all parameters from tlc to caffe
	params = tlcNet.params
	for i in range(1, len(tlcNet.layers)):
		tlcLayer = tlcNet.layers[i]
		for l in caffeNet.layers:
			if l.name == tlcLayer.name:
				caffeLayer = l
		bundle = tlcLayer.bundle
		if type(bundle) is ConvolveBundle:
			# source blob
			weights = params.params[bundle.weights]
			# target blobs
			weightBlob = caffeLayer.blobs.add()
			biasBlob = caffeLayer.blobs.add()
			if bundle.sharing is not None and not bundle.sharing[-1]:
				local = True
			else:
				local = False
			kernelSize = reduce(lambda x,y: x*y, bundle.geo.dimKernel)

			if local:
				height_out = 1 + (bundle.geo.dimInput[-2] + caffeLayer.local_param.pad * 2 - bundle.geo.dimKernel[-2]) / caffeLayer.local_param.stride
				width_out = 1 + (bundle.geo.dimInput[-1] + caffeLayer.local_param.pad * 2 - bundle.geo.dimKernel[-1]) / caffeLayer.local_param.stride
				# local convolution
				
				weightBlob.num = caffeLayer.local_param.num_output
				weightBlob.channels = 1
				weightBlob.height = reduce(lambda x,y: x*y, bundle.geo.dimKernel)
				weightBlob.width = height_out * width_out

				biasBlob.num = 1
				biasBlob.channels = 1
				biasBlob.height = caffeLayer.local_param.num_output
				biasBlob.width = weightBlob.width

				index = 0
				M_ = caffeLayer.local_param.num_output
				N_ = height_out * width_out
				# numKernel = M_ * N_
				kernels = np.ndarray(shape=(M_, N_, kernelSize), dtype=float)
				for i in range(M_):
					for j in range(N_):
						kernels[i][j] = weights[index:index+kernelSize]
						biasBlob.data.append(weights[index+kernelSize])
						index += kernelSize + 1

				for i in range(M_):
					weightBlob.data.extend(kernels[i].T.flatten().tolist())

			else:
				# global convolution
				# numKernel equals to num of feature maps
				numKernel = caffeLayer.convolution_param.num_output 
				weightBlob.num = numKernel
				if len(bundle.geo.dimKernel) == 2:
					weightBlob.channels = 1
				else:
					weightBlob.channels = bundle.geo.dimKernel[0]
				weightBlob.height = caffeLayer.convolution_param.kernel_size
				weightBlob.width = caffeLayer.convolution_param.kernel_size

				biasBlob.num = 1
				biasBlob.channels = 1
				biasBlob.height = 1
				biasBlob.width = numKernel

				# start copy the parameters
				index = 0
				for i in range(numKernel):
					weightBlob.data.extend(weights[index:index+kernelSize])
					biasBlob.data.append(weights[index+kernelSize])
					index += kernelSize + 1
			# print weightBlob
			# print biasBlob
			
			# print "%s: %s * %s" % (tlcLayer.name, numKernel, kernelSize)

		elif type(bundle) is FullBundle:
			weights = params.params[bundle.weights]
			weightBlob = caffeLayer.blobs.add()
			weightBlob.data.extend(weights)
			weightBlob.num = 1
			weightBlob.channels = 1
			weightBlob.height = tlcLayer.dimOutput
			weightBlob.width = len(weights) / weightBlob.height

			biases = params.params[tlcLayer.biases]
			biasBlob = caffeLayer.blobs.add()
			biasBlob.data.extend(biases)
			biasBlob.num = 1
			biasBlob.channels = 1
			biasBlob.height = 1
			biasBlob.width = tlcLayer.dimOutput

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "%s TLCNetFile" % sys.argv[0]
		exit()

	filename = sys.argv[1]
	prefix = os.path.splitext(os.path.basename(filename))[0]

	fin = open(filename)
	print "[%s] Start loading TLC model" % time.asctime()
	tlcNet = NeuralNetwork.parseNet(fin)
	tlcNet.params.loadBlobs(fin)
	print "[%s] Loading finished" % time.asctime()
	
	protofile = "%s.prototxt" % prefix
	caffeNet = convert(tlcNet)
	with open(protofile, 'w') as f:
		f.write(str(caffeNet))
	print "[%s] Proto file: %s" % (time.asctime(), protofile)
	
	if tlcNet.layers[1].bundle.weights is not None:	
		binaryfile = "%s.caffemodel" % prefix
		convertBlobs(tlcNet, caffeNet)
		with open(binaryfile, 'wb') as f:
			f.write(caffeNet.SerializeToString())
		print "[%s] Binary model file: %s" % (time.asctime(), binaryfile)

	print "[%s] Finished" % time.asctime()
