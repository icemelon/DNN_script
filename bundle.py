from const import Const

# bundle is the input function for each layer
# could be convolve, fully, max pool, responive norm
class Bundle(object):
	def __init__(self, inputLayer):
		self.input = inputLayer

	@staticmethod
	def parse(inputLayer, tokens):
		bundleType = tokens.pop()
		if bundleType == "max" or bundleType == "response":
			bundleType += " %s" % tokens.pop()
		attrs = {}
		# NOTE: startswith could handle empty string case
		if tokens.content.startswith('{'):
			detail = tokens.pop().strip('{}')
			for item in detail.split(';'):
				item = item.strip()
				if len(item) == 0: continue
				key, val = item.split('=')
				attrs[key.strip()] = Const.parseValueOrArray(val)

		if bundleType == "all":
			bundle = FullBundle(inputLayer, attrs)
		elif bundleType == "convolve":
			bundle = ConvolveBundle(inputLayer, attrs)
		elif bundleType == "max pool":
			bundle = MaxPoolBundle(inputLayer, attrs)
		elif bundleType == "response norm":
			bundle = ResponseNormBundle(inputLayer, attrs)
		else:
			raise Exception('Wrong bundle type %s' % bundleType)

		return bundle

class FullBundle(Bundle):
	def __init__(self, inputLayer, attrs):
		super(FullBundle, self).__init__(inputLayer)
		self.weights = attrs.get("Weights", None)

	def __str__(self):
		if self.weights is None:
			return "  from %s all;" % self.input.name
		else:
			return "  from %s all { Weights = %s; }" % (self.input.name, Const.tostr(self.weights))

class ConvolveBundle(Bundle):
	def __init__(self, inputLayer, attrs):
		super(ConvolveBundle, self).__init__(inputLayer)
		self.geo = ConvolveGeometry(attrs)
		self.mapCount = attrs['MapCount']
		self.sharing = attrs['Sharing'] if 'Sharing' in attrs \
			else [False] * self.geo.dim
		self.weights = attrs['Weights'] if 'Weights' in attrs else None

	def computeDimOutput(self):
		dimOutput = self.geo.dimOutput
		dimOutput[0] *= self.mapCount
		return dimOutput

	def __str__(self):
		s = '  from %s convolve {\n' % self.input.name
		pad = ' '*4
		s += '%s\n' % self.geo.__str__(pad)
		s += pad + 'Sharing = %s;\n' % Const.tostr(self.sharing)
		s += pad + 'MapCount = %s;\n' % Const.tostr(self.mapCount)
		if self.weights is not None:
			s += pad + 'Weights = %s\n' % Const.tostr(self.weights)
		s += '  }'
		return s

class MaxPoolBundle(Bundle):
	def __init__(self, inputLayer, attrs):
		super(MaxPoolBundle, self).__init__(inputLayer)
		self.geo = ConvolveGeometry(attrs)

	def computeDimOutput(self):
		return self.geo.dimOutput

	def __str__(self):
		s = '  from %s max pool {\n' % self.input.name
		pad = ' '*4
		s += '%s\n' % self.geo.__str__(pad)
		s += '  }'
		return s

class ResponseNormBundle(Bundle):
	def __init__(self, inputLayer, attrs):
		super(ResponseNormBundle, self).__init__(inputLayer)
		self.geo = ConvolveGeometry(attrs)
		self.alpha = attrs['Alpha']
		self.beta = attrs['Beta']
		self.offset = attrs['Offset']
		self.avgOverFullKernel = attrs['AvgOverFullKernel']

	def computeDimOutput(self):
		return self.geo.dimOutput

	def __str__(self):
		s = '  from %s response norm {\n' % self.input.name
		pad = ' ' *4
		s += '%s\n' % self.geo.__str__(pad)
		s += pad + 'Alpha = %s;\n' % Const.tostr(self.alpha)
		s += pad + 'Beta = %s;\n' % Const.tostr(self.beta)
		s += pad + 'Offset = %s;\n' % Const.tostr(self.offset)
		s += pad + 'AvgOverFullKernel = %s;\n' % Const.tostr(self.avgOverFullKernel)
		s += '  }'
		return s

class ConvolveGeometry(object):
	def __init__(self, attrs):
		self.dimInput = attrs['InputShape']
		self.dimKernel = attrs['KernelShape']
		self.dim = len(self.dimKernel)
		if 'Stride' in attrs:
			self.stride = attrs['Stride']
		else:
			self.stride = [1] * self.dim
		if 'Padding' in attrs:
			self.padding = attrs['Padding']
		else:
			self.padding = [False] * self.dim

	@property
	def dimOutput(self):
		dimOutput = []
		for i in range(self.dim):
			nInput = self.dimInput[i]
			if self.padding[i]:
				nInput += self.dimKernel[i] - 1
			nOutput = (nInput - self.dimKernel[i]) / self.stride[i] + 1
			dimOutput.append(nOutput)
		# print dimOutput
		return dimOutput

	def __str__(self, pad=''):
		s = pad + 'InputShape = %s;\n' % Const.tostr(self.dimInput)
		s += pad + 'KernelShape = %s;\n' % Const.tostr(self.dimKernel)
		s += pad + 'Stride = %s;\n' % Const.tostr(self.stride)
		s += pad + 'Padding = %s;' % Const.tostr(self.padding)
		return s
