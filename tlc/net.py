import time
import operator

from layer import *
from param import Param
from tokenparser import FileTokenParser

OPS = {
	"+": operator.add,
	"-": operator.sub,
	"*": operator.mul,
	'/': operator.div,
}

class NeuralNetwork(object):
	def __init__(self, params=None):
		self.layers = []
		self.params = Param() if params is None else params
	
	def output(self):
		return '\n'.join(l.output() for l in self.layers)

	def outputParams(self, fout, fin=None):
		self.params.outputParams(fout, fin)

	def findLayerByName(self, name):
		ret = None
		for l in self.layers:
			if l.name == name:
				ret = l
				break
		return ret

	def verify(self):
		# first check if there is any dangling constant
		for key in self.params.values:
			val = self.params.values[key]
			if val is None:
				print "Error: find dangling constant: %s" % key
				return False
		return True

		for l in self.layers:
			if 'geo' in dir(l):
				dimOutput = l.computeDimOutput()
				if l.geo.dimInput[0] < l.geo.dimKernel[0]:
					"Error in layer %s" % l.name
					print "Input size and kernel size doesn't match: %s(input) vs %s(kernel)" % (l.geo.dimInput, l.geo.dimKernel)
					return False
				if dimOutput != l.dimOutput:
					print "Error in layer %s" % l.name
					print "Output size is incorrect: %s vs %s(computed)" % (l.dimOutput, dimOutput)
					return False
				if l.inputLayer.dimOutput != l.geo.dimInput:
					print "Error in layer %s" % l.name
					print "Input size doesn't match: %s(output) vs %s(input)" % (l.inputLayer.dimOutput, l.geo.dimInput)
					return False
				
		return True

	# num: split at num-th layer from the beginning
	# return: (bottom NN, top NN)
	def split(self, num):
		# init topNN and its input
		topNN = NeuralNetwork(self.params)
		dimInput = self.layers[num - 1].dimOutput
		if type(dimInput) is list:
			dimInput = reduce(lambda x,y: x*y, dimInput)
		topInputLayer = InputLayer("vector", dimInput, self.params)
		topNN.layers.append(topInputLayer)

		# add the rest of top layers into topNN
		for i in range(num, len(self.layers)):
			layer = self.layers[i]
			layer.removeParams()
			topNN.layers.append(layer)
		# change the input of first hidden layer in topNN
		topNN.layers[1].bundle.input = topInputLayer

		# remove top layers from bottomNN
		for _ in range(len(self.layers) - num):
			self.layers.pop()

		# change last layer of bottomNN
		layer = self.layers.pop()
		attrs = {"Biases": layer.biases}
		newLayer = OutputLayer(layer.name, layer.dimOutput, self.params, layer.outputFunc, layer.bundle, attrs)
		self.layers.append(newLayer)

		return (self, topNN)

	# num: num of last layers to remove
	def removeLayers(self, num):
		for _ in range(num):
			layer = self.layers.pop()
			layer.removeParams()
		# change last layer into output layer
		layer = self.layers.pop()
		attrs = {"Biases": layer.biases}
		newLayer = OutputLayer(layer.name, layer.dimOutput, self.params, layer.outputFunc, layer.bundle, attrs)
		self.layers.append(newLayer)

	def update(self, layerName, mod):
		layer = None
		for i in range(len(self.layers)):
			l = self.layers[i]
			if l.name == layerName:
				layer = l
				break
		if layer is None:
			raise Exception('Cannot find layer %s' % layerName)
		if 'Stride' in mod:
			assert 'geo' in dir(layer)
			val, op = mod['Stride']
			if op == '=':
				layer.geo.stride[1] = val
				layer.geo.stride[2] = val
			else:
				layer.geo.stride[1] = OPS[op](layer.geo.stride[1], val)
				layer.geo.stride[2] = OPS[op](layer.geo.stride[2], val)
			layer.dimOutput = layer.computeDimOutput()
		if 'MapCount' in mod:
			assert isinstance(layer, ConvolveLayer)
			val, op = mod['MapCount']
			if op == '=':
				layer.mapCount = val
			else:
				layer.mapCount = OPS[op](layer.mapCount, val)
			layer.dimOutput = layer.computeDimOutput()

		for j in range(i, len(self.layers)):
			l = self.layers[j]
			if 'geo' in dir(l):
				l.geo.dimInput = l.inputLayer.dimOutput
				if isinstance(l, ConvolveLayer):
					l.geo.dimKernel[0] = l.inputLayer.dimOutput[0]
				l.dimOutput = l.computeDimOutput()
				if layer.geo.dimKernel[0] > layer.geo.dimInput[0]:
					layer.geo.dimKernel[0] = layer.geo.dimInput[0]

	@staticmethod
	def parseNet(fin, loadParams=False):
		tokens = FileTokenParser(fin)

		net = NeuralNetwork()
		while True:
			token = tokens.peek()
			if token == None:
				break
			elif token == 'const':
				net.params.parseConst(tokens)
			else:
				layer = Layer.parse(tokens, net)
				net.layers.append(layer)
				# print layer
				if type(layer) is OutputLayer and not loadParams:
					# only parse to output layer
					break
		return net
