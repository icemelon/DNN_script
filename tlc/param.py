import util

class Param(object):

	def __init__(self):
		self.params = {}
	
	def parseConst(self, tokens):
		tokens.pop() # skip "const"
		if tokens.startswith('{'):
			desc = tokens.pop().strip('{}')
			for item in desc.split(';'):
				item = item.strip()
				if len(item) == 0: continue
				key, val = item.split('=')
				self.params[key.strip()] = util.parseValueOrArray(val)
		else:
			key = tokens.pop()
			tokens.pop() # skip '='
			val = tokens.pop(separators=';')
			self.params[key.strip()] = util.parseLargeArray(val)

	def loadBlobs(self, fin):
		var = None
		hexFormat = False
		while True:
			line = fin.readline()
			if line is None or len(line) == 0: break
			line = line.strip()
			if len(line) == 0:
				continue
			elif line.startswith("const"):
				tokens = line.split()
				var = tokens[1]
				self.params[var] = []
				continue
			elif line.startswith('floats_from_bytes'):
				hexFormat = True
				continue
			elif ']' in line:
				var = None
				hexFormat = False
				continue

			# now starts to parsing values
			if '//' in line: line = line[:line.index('//')]
			# print line
			if hexFormat:
				if ')' in line: continue
				self.params[var].extend([util.parseHex(s) for s in line.split(' ')])
			else:
				self.params[var].extend([eval(s) for s in line.split(',') if s.strip()])

	def parseParam(self, s):
		if s in self.params:
			return self.params[s]
		try:
			val = util.parseValueOrArray(s)
		except Exception:
			val = s.strip() # return symbol only
			self.params[val] = None # put a holder in the params lookup
		return val

	def remove(self, var):
		if var in self.params:
			del self.params[var]
			return True
		return False

	def output(self, val):
		if type(val) is bool:
			return 'true' if val else 'false'
		elif type(val) is list:
			return '[' + ', '.join(self.output(x) for x in val) + ']'
		else:
			return str(val)

	# fin: parsing the rest of input file (large blobs part)
	def outputParams(self, fout, fin=None):
		for key in self.params:
			val = self.params[key]
			if type(val) is list:
				fout.write("const %s = %s;\n" % (key, val))

		if fin is not None:
			while True:
				line = fin.readline()
				if not line: break
				line = line.strip('\n')
				if len(line) == 0: continue
				if not line.startswith("const"): continue

				var = line.split()[1]
				# check if this const is required
				if var in self.params:
					fout.write("\n%s\n" % line)
					end = False
					while not end:
						line = fin.readline().strip('\n')
						fout.write("%s\n" % line)
						if "]" in line: end = True
				else:
					# skip this const
					end = False
					while not end:
						line = fin.readline().strip('\n')
						if "]" in line: end = True

