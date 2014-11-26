import os
import sys
import math
import time

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
		imageSize = int(math.sqrt(inputLayer.dimOutput / 3))
		caffeNet.input_dim.append(3)
		caffeNet.input_dim.append(imageSize)
		caffeNet.input_dim.append(imageSize)

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
			weights = params.params[bundle.weights]
			weightBlob = caffeLayer.blobs.add()
			weightBlob.num = caffeLayer.convolution_param.num_output
			if len(bundle.geo.dimKernel) == 2:
				weightBlob.channels = 1
			else:
				weightBlob.channels = bundle.geo.dimKernel[0]
			weightBlob.height = caffeLayer.convolution_param.kernel_size
			weightBlob.width = caffeLayer.convolution_param.kernel_size

			biasBlob = caffeLayer.blobs.add()
			biasBlob.num = 1
			biasBlob.channels = 1
			biasBlob.height = 1
			biasBlob.width = weightBlob.num

			kernel_num = weightBlob.channels * weightBlob.height * weightBlob.width
			index = 0
			for i in range(weightBlob.num):
				weightBlob.data.extend(weights[index:index+kernel_num])
				biasBlob.data.append(weights[index+kernel_num])
				index += kernel_num + 1

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
	
	binaryfile = "%s.caffemodel" % prefix
	convertBlobs(tlcNet, caffeNet)
	with open(binaryfile, 'wb') as f:
		f.write(caffeNet.SerializeToString())
	print "[%s] Binary model file: %s" % (time.asctime(), binaryfile)
