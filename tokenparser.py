SEPERATOR = " {;"

class TokenParser(object):
	def __init__(self, string):
		self.content = string[:].strip()
		self.end = False
		self._token = None # for cache use

	def _parse(self, sep):
		if self.end: return

		if self.content[0] == '{':
			index = 0
			stack = 0
			while True:
				if self.content[index] == '{':
					stack += 1
				elif self.content[index] == '}':
					stack -= 1
				index += 1
				if stack == 0:
					break
			self._token = self.content[:index]
			self.content = self.content[index:].strip()
		else:
			token = ''
			if self.content[0] == '[':
				index = 0
				stack = 0
				while True:
					if self.content[index] == '[':
						stack += 1
					elif self.content[index] == ']':
						stack -= 1
					index += 1
					if stack == 0:
						break
				token = self.content[:index]
				self.content = self.content[index:]
				
			index = -1
			for c in sep:
				# print c
				i = self.content.find(c)
				# corner case: a={}
				# we look for second occurrence
				if c == '{' and i > 0 and self.content[i-1] == '=':
					i = self.content[i+1:].find('{')
				if i < 0: continue
				if index < 0 or i < index:
					index = i
			# last token
			if index < 0: index = len(self.content)

			token += self.content[:index]
			self._token = token
			self.content = self.content[index+1:].strip()

		if len(self.content) == 0:
			self.end = True

	def pop(self, sep=SEPERATOR):
		if self._token is None:
			self._parse(sep)
		token = self._token
		self._token = None
		return token

	# get next token, but didn't pop up it
	def peek(self, sep=SEPERATOR):
		if self._token is None:
			self._parse(sep)
		return self._token