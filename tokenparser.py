SEPARATOR = " {;"

class TokenParser(object):
	def pop(self, separators=SEPARATOR):
		pass

	def peek(self, separators=SEPARATOR):
		pass

	def startswith(self, prefix):
		pass

class StringTokenParser(TokenParser):
	def __init__(self, string):
		self._content = string[:].strip()
		self.end = False
		self._token = None # current token
		self._sep = None # current separator

	def _parse(self, separators):
		if self.end: return

		if len(self._content) == 0:
			self.end = True
			return

		if self._content[0] == '{':
			index = 0
			stack = 0
			while True:
				if self._content[index] == '{':
					stack += 1
				elif self._content[index] == '}':
					stack -= 1
				index += 1
				if stack == 0:
					break
			self._token = self._content[:index]
			self._sep = '}'
			self._content = self._content[index:].strip()
			return

		token = ''
		if self._content[0] == '[':
			index = 0
			stack = 0
			while True:
				if self._content[index] == '[':
					stack += 1
				elif self._content[index] == ']':
					stack -= 1
				index += 1
				if stack == 0:
					break
			token = self._content[:index]
			self._content = self._content[index:]
			
		index = -1
		sep = None
		for c in separators:
			# print c
			i = self._content.find(c)
			# corner case: a={}
			# we look for second occurrence
			if c == '{' and i > 0 and self._content[i-1] == '=':
				i = self._content[i+1:].find('{')
			if i < 0: continue
			if index < 0 or i < index:
				index = i
				sep = c
		# last token
		if index < 0:
			index = len(self._content)
			sep = "<EOF>"

		token += self._content[:index]
		self._token = token.strip()
		self._sep = sep
		if sep == '{':
			self._content = self._content[index:].strip()
		else:
			self._content = self._content[index+1:].strip()

	def pop(self, separators=SEPARATOR):
		if self._token is None:
			self._parse(separators)
		token = self._token
		self._token = None
		return token

	# get next token, but didn't pop up it
	def peek(self, separators=SEPARATOR):
		if self._token is None:
			self._parse(separators)
		return self._token

	def startswith(self, prefix):
		return self._content.startswith(prefix)

class FileTokenParser(TokenParser):
	def __init__(self, fin):
		self._content = ""
		self.end = False
		self._token = None # for cache use
		
		self.fin = fin
		self.fileEnds = False
		self.readline()

	def readline(self):
		if self.fileEnds:
			return
		line = self.fin.readline()
		if not line:
			self.fileEnds = True
			return
		if '//' in line:
			line = line[:line.index('//')]
		self._content += " " + line.strip()
		self._content = self._content.strip()

	def _parse(self, separators):
		if self.end: return

		if len(self._content) == 0:
			self.end = True
			return

		if self._content[0] == '{':
			index = 0
			stack = 0
			while True:
				if self._content[index] == '{':
					stack += 1
				elif self._content[index] == '}':
					stack -= 1
				index += 1
				if index == len(self._content):
					self.readline()
				if stack == 0:
					break
			self._token = self._content[:index]
			self._sep = '}'
			self._content = self._content[index:].strip()
			return

		token = ''
		if self._content[0] == '[':
			index = 0
			stack = 0
			while True:
				if self._content[index] == '[':
					stack += 1
				elif self._content[index] == ']':
					stack -= 1
				index += 1
				if index == len(self._content):
					self.readline()
				if stack == 0:
					break
			token = self._content[:index]
			self._content = self._content[index:]
			
		index = -1
		sep = None
		for c in separators:
			# print c
			i = self._content.find(c)
			# corner case: a={}
			# we look for second occurrence
			if c == '{' and i > 0 and self._content[i-1] == '=':
				i = self._content[i+1:].find('{')
			if i < 0: continue
			if index < 0 or i < index:
				index = i
				sep = c
		# last token
		if index < 0:
			index = len(self._content)
			sep = "<EOF>"

		token += self._content[:index]
		self._token = token.strip()
		self._sep = sep
		if sep == '{':
			self._content = self._content[index:].strip()
		else:
			self._content = self._content[index+1:].strip()

	def pop(self, separators=SEPARATOR):
		while not self.fileEnds and len(self._content) == 0:
			self.readline()
		if self._token is None:
			self._parse(separators)
		token = self._token
		self._token = None
		return token

	# get next token, but didn't pop up it
	def peek(self, separators=SEPARATOR):
		while not self.fileEnds and len(self._content) == 0:
			self.readline()
		if self._token is None:
			self._parse(separators)
		return self._token

	def startswith(self, prefix):
		while not self.fileEnds and len(self._content) == 0:
			self.readline()
		return self._content.startswith(prefix)
