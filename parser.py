import os
import re
import sys
import time
import argparse

from tlc import net

def isnum(expr):
	p = re.compile('\d+(\.\d+)?')
	if p.match(expr):
		return True
	else:
		return False

def log(msg):
	timestr = time.strftime("%m/%d/%Y %H:%M:%S", time.localtime())
	print "%s %s" % (timestr, msg)

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(description='Parse/Convert TLC NN file')
	argparser.add_argument('input', type=file, help='input file name (.nn)')
	# argparser.add_argument('-v', '--verify', action='store_true', dest='verify', help='Verify NN definition')
	argparser.add_argument('-rm', '--remove', type=int, default=0, help='Remove last N layers')
	argparser.add_argument('-m', '--modify', help='Modification description file')
	argparser.add_argument('-o', '--output', type=argparse.FileType('w'), help='output file name (.nn)')
	argparser.add_argument('--stats', action='store_true')
	try:
		args = argparser.parse_args()
	except:
		exit()

	begin = time.time()
	log("Starts to parse neural network")
	net = net.NeuralNetwork.parseNet(args.input)
	end = time.time()
	log("Parsing finishes (%.1f s)" % (end-begin))
	# if args.verify:
	# 	if nn.verify():
	# 		print '%s is legal' % args.input
	# 	else:
	# 		print '%s is not legal' % args.input
	# 	exit()
	
	if args.modify is not None:
		with open(args.modify) as f:
			content = f.readlines()
		i = 0
		while i < len(content):
			line = content[i].strip()
			i += 1
			tokens = line.split(':')
			assert(tokens[0] == 'Layer')
			layerName = tokens[1].strip()
			mod = {}
			while i < len(content):
				line = content[i].strip()
				if line.startswith('Layer'):
					break
				i += 1
				tokens = line.split(':')
				expr = tokens[1].strip()
				if isnum(expr):
					op = '='
					val = eval(expr)
				else:
					op = expr[0]
					val = eval(expr[1:])
				mod[tokens[0].strip()] = (val, op)
			net.update(layerName, mod)
		# if not nn.verify():
		# 	exit()

	if args.remove > 0:
		net.removeLayers(args.remove)

	if args.output:
		begin = time.time()
		log("Starts to output neural network")
		
		args.output.write("%s\n" % net.output())
		net.outputParams(args.output, args.input)

		end = time.time()
		log("Output finishes (%.1f s)" % (end-begin))

		args.input.close()
		args.output.close()
	elif args.stats:
		net.stats()
