
# bundle is the input function for each layer
# could be convolve, fully, max pool, responive norm
class Bundle(object):
	def __init__(self, inputLayer, params):
		self.input = inputLayer
		self.params = params

	def removeParams(self):
		pass

	def output(self):
		pass

	@staticmethod
	def parse(inputLayer, tokens, params):
		bundleType = tokens.pop(separators='{;')
		attrs = {}
		# check if there are more attributes
		if tokens._sep != ';':
			detail = tokens.pop().strip('{}')
			for item in detail.split(';'):
				item = item.strip()
				if len(item) == 0: continue
				key, val = item.split('=')
				attrs[key.strip()] = params.parseParam(val)

		if bundleType == "all":
			bundle = FullBundle(inputLayer, params, attrs)
		elif bundleType == "convolve":
			bundle = ConvolveBundle(inputLayer, params, attrs)
		elif bundleType == "max pool":
			bundle = MaxPoolBundle(inputLayer, params, attrs)
		elif bundleType == "response norm":
			bundle = ResponseNormBundle(inputLayer, params, attrs)
		else:
			raise Exception('Wrong bundle type %s' % bundleType)

		return bundle

class FullBundle(Bundle):
	def __init__(self, inputLayer, params, attrs):
		super(FullBundle, self).__init__(inputLayer, params)
		self.weights = attrs.get("Weights", None)

	def removeParams(self):
		if self.weights is not None:
			self.params.remove(self.weights)
			self.weights = None

	def output(self):
		if self.weights is None:
			return "  from %s all;" % self.input.name
		else:
			return "  from %s all { Weights = %s; }" % (self.input.name, self.params.output(self.weights))

class ConvolveBundle(Bundle):
	def __init__(self, inputLayer, params, attrs):
		super(ConvolveBundle, self).__init__(inputLayer, params)
		self.geo = ConvolveGeometry(params, attrs)
		self.mapCount = attrs['MapCount']
		self.sharing = attrs['Sharing'] if 'Sharing' in attrs else None
		self.weights = attrs['Weights'] if 'Weights' in attrs else None

	def removeParams(self):
		if self.weights is not None:
			self.params.remove(self.weights)
			self.weights = None

	# current not working		
	# def computeDimOutput(self):
	# 	dimOutput = self.geo.dimOutput
	# 	dimOutput[0] *= self.mapCount
	# 	return dimOutput

	def output(self):
		s = '  from %s convolve {\n' % self.input.name
		pad = ' '*4
		s += '%s\n' % self.geo.output(pad)
		if self.sharing is not None:
			s += pad + 'Sharing = %s;\n' % self.params.output(self.sharing)
		s += pad + 'MapCount = %s;\n' % self.params.output(self.mapCount)
		if self.weights is not None:
			s += pad + 'Weights = %s;\n' % self.params.output(self.weights)
		s += '  }'
		return s

class MaxPoolBundle(Bundle):
	def __init__(self, inputLayer, params, attrs):
		super(MaxPoolBundle, self).__init__(inputLayer, params)
		self.geo = ConvolveGeometry(params, attrs)

	def computeDimOutput(self):
		return self.geo.dimOutput

	def output(self):
		s = '  from %s max pool {\n' % self.input.name
		pad = ' '*4
		s += '%s\n' % self.geo.output(pad)
		s += '  }'
		return s

class ResponseNormBundle(Bundle):
	def __init__(self, inputLayer, params, attrs):
		super(ResponseNormBundle, self).__init__(inputLayer, params)
		self.geo = ConvolveGeometry(params, attrs)
		self.alpha = attrs['Alpha']
		self.beta = attrs['Beta']
		self.offset = attrs['Offset']
		self.avgOverFullKernel = attrs['AvgOverFullKernel']

	def computeDimOutput(self):
		return self.geo.dimOutput

	def output(self):
		s = '  from %s response norm {\n' % self.input.name
		pad = ' ' *4
		s += '%s\n' % self.geo.output(pad)
		s += pad + 'Alpha = %s;\n' % self.params.output(self.alpha)
		s += pad + 'Beta = %s;\n' % self.params.output(self.beta)
		s += pad + 'Offset = %s;\n' % self.params.output(self.offset)
		s += pad + 'AvgOverFullKernel = %s;\n' % self.params.output(self.avgOverFullKernel)
		s += '  }'
		return s

class ConvolveGeometry(object):
	def __init__(self, params, attrs):
		self.params = params
		self.dimInput = attrs['InputShape']
		self.dimKernel = attrs['KernelShape']
		self.dim = len(self.dimKernel)
		self.stride = attrs['Stride'] if 'Stride' in attrs else None
		self.padding = attrs['Padding'] if 'Padding' in attrs else None

	@property
	def dimOutput(self):
		dimOutput = []
		for i in range(self.dim):
			nInput = self.dimInput[i]
			if self.padding[i]:
				nInput += self.dimKernel[i] - 1
			nOutput = (nInput - self.dimKernel[i]) / self.stride[i] + 1
			dimOutput.append(nOutput)
		return dimOutput

	def output(self, pad=''):
		items = []
		items.append('InputShape = %s;' % self.params.output(self.dimInput))
		items.append('KernelShape = %s;' % self.params.output(self.dimKernel))

		if self.stride is not None:
			items.append('Stride = %s;' % self.params.output(self.stride))
		if self.padding is not None:
			items.append('Padding = %s;' % self.params.output(self.padding))
		sep = "\n" + pad
		s = pad + sep.join(items)
		return s
