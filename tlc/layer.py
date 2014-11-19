import util
from tokenparser import *
from bundle import *

DefaultHiddenLayerFunction = 'sigmoid'
DefaultOutputLayerFunction = 'softmax'

class Layer(object):
	def __init__(self, name, dimOutput, params):
		self.name = name
		self.params = params
		self.dimOutput = dimOutput

	def removeParams(self):
		pass

	def output(self):
		pass

	def computeDimOutput(self):
		return self.dimOutput

	def dimOutput_to_str(self):
		if type(self.dimOutput) is list:
			return str(self.dimOutput)
		else:
			return "[%s]" % self.dimOutput

	@staticmethod
	def parse(tokens, net):
		layerType = tokens.pop()
		name = tokens.pop()
		dimOutput = util.parseValueOrArray(tokens.pop())

		if layerType == 'input':
			return InputLayer(name, dimOutput, net.params)

		token = tokens.pop()
		if 'from' not in token:
			outputFunc = token
			token = tokens.pop()
		else:
			outputFunc = None

		attrs = {}
		if token[0] == '{':
			tokens = StringTokenParser(token.strip('{}'))
			while not tokens.startswith("from"):
				desc = tokens.pop(separators=";")
				key, val = desc.split('=')
				attrs[key.strip()] = net.params.parseParam(val)
			tokens.pop() # skip "from"
		else:
			assert token == "from"

		inputLayerName = tokens.pop()
		inputLayer = net.findLayerByName(inputLayerName)
		# print "input: %s" % inputLayer
		if inputLayer is None:
			raise Exception('Cannot find input layer: %s' % inputLayerName)

		bundle = Bundle.parse(inputLayer, tokens, net.params)

		if layerType == "hidden":
			return HiddenLayer(name, dimOutput, net.params, outputFunc, bundle, attrs)
		elif layerType == "output":
			return OutputLayer(name, dimOutput, net.params, outputFunc, bundle, attrs)
		else:
			raise Exception('Wrong layer type: %s' % layerType)

class InputLayer(Layer):
	def __init__(self, name, dimOutput, params):
		super(InputLayer, self).__init__(name, dimOutput, params)

	def output(self):
		return 'input %s %s;' % (self.name, self.dimOutput_to_str())

class HiddenLayer(Layer):
	def __init__(self, name, dimOutput, params, outputFunc, bundle, attrs):
		super(HiddenLayer, self).__init__(name, dimOutput, params)
		# output function can be rlinear, sigmoid, softmax
		self.outputFunc = DefaultHiddenLayerFunction if outputFunc is None else outputFunc
		# bundle is the input function
		# could be convolve, fully, max pool, responive norm
		self.bundle = bundle
		self.biases = attrs.get("Biases", None)

	def removeParams(self):
		if self.biases is not None:
			self.params.remove(self.biases)
			self.biases = None
		self.bundle.removeParams()

	def output(self):
		s = "hidden %s %s %s {\n" % (self.name, self.dimOutput_to_str(), self.outputFunc)
		if self.biases is not None:
			s += "  Biases = %s;\n" % self.params.output(self.biases)
		s += "%s\n}" % self.bundle.output()
		return s

class OutputLayer(Layer):
	def __init__(self, name, dimOutput, params, outputFunc, bundle, attrs):
		super(OutputLayer, self).__init__(name, dimOutput, params)
		self.outputFunc = DefaultHiddenLayerFunction if outputFunc is None else outputFunc
		self.bundle = bundle
		self.biases = attrs.get("Biases", None)

	def removeParams(self):
		if self.biases is not None:
			self.params.remove(self.biases)
			self.biases = None
		self.bundle.removeParams()

	def output(self):
		s = "output %s %s %s {\n" % (self.name, self.dimOutput_to_str(), self.outputFunc)
		if self.biases is not None:
			s += "  Biases = %s;\n" % self.params.output(self.biases)
		s += "%s\n}" % self.bundle.output()
		return s

