import time
import operator

from const import Const
from layer import *
from tokenparser import FileTokenParser

OPS = {
	"+": operator.add,
	"-": operator.sub,
	"*": operator.mul,
	'/': operator.div,
}

class NeuralNetwork(object):
	def __init__(self):
		self.layers = []

	def parse(self, content):
		if content.startswith('const'):
			Const.parseConst(content)
		else:
			layer = Layer.parse(content, self.layers)
			self.layers.append(layer)
	
	def __str__(self):
		return '\n'.join(str(l) for l in self.layers)

	def findLayerByName(self, name):
		ret = None
		for l in self.layers:
			if l.name == name:
				ret = l
				break
		return ret

	def verify(self):
		# first check if there is any dangling constant
		for key in Const.values:
			val = Const.values[key]
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
	def parseNN(fin):
		tokens = FileTokenParser(fin)

		nn = NeuralNetwork()
		while True:
			token = tokens.peek()
			if token == None:
				break
			elif token == 'const':
				Const.parse(tokens)
			else:
				layer = Layer.parse(tokens, nn)
				nn.layers.append(layer)
				if type(layer) is OutputLayer:
					# only parse to output layer
					break
		return nn
