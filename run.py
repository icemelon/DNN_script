import os
import argparse

import hdp
from config import *
from trainer import *
from job import JobManager
from logger import Logger

def runLocal(args):
	if args.tlc is None:
		args.tlc = DEFAULT_LOCAL_TLC_PATH
	args.dataset = os.path.abspath(args.dataset)
	args.logfile = os.path.abspath(args.logfile)
	
	print 'TLC path: %s' % args.tlc
	print 'Socket: %s' % args.socket

	jobManager = JobManager(args.socket, args.tlc)
	logger = Logger(args.logfile)

	# switch to dataset directory
	os.chdir(args.dataset)

	if 'Shared' in logger.headers:
		trainer = SharedTrainer(logger, args.dataset, jobManager)
	else:
		trainer = RegularTrainer(logger, args.dataset, jobManager)
	print
	print trainer.summary()
	trainer.train()

def runHDP(args):
	if args['tlc'] is None:
		args['tlc'] = DEFAULT_HDP_TLC_PATH

	workdir = DEFAULT_HDP_ROOT_DIR
	script = os.path.relpath(__file__)
	# now hdp jobs are node exclusive 
	nodes = (args['socket'] + 1) / 2
	args['socket'] = nodes * 2 # each node has 2 GPUs/sockets

	command = "python %s %s %s -s %s --tlc %s --env local" % \
		(script, args['dataset'], args['logfile'], args['socket'], args['tlc'])
	stdout = os.path.splitext(args['logfile'])[0] + ".out"
	stderr = os.path.splitext(args['logfile'])[0] + ".err"

	dataset = os.path.basename(args['dataset'])
	threadName =  os.path.splitext(os.path.basename(args['logfile']))[0]
	jobName = "%s_%s" % (dataset, threadName)

	print 'JobTemplate: %s, Node: %s, Socket: %s, WorkDir: %s' \
		% (args['jobtemp'], nodes, args['socket'], workdir)

	hdp.execute(args['jobtemp'], jobName, nodes, workdir, command, stdout, stderr)

	return jobName

if __name__ == '__main__':
	argParser = argparse.ArgumentParser(description='Using TLC to train neural networks')
	argParser.add_argument('dataset', help='Working directory that contains dataset')
	argParser.add_argument('logfile', help='Log file (header should be initialed)')
	argParser.add_argument('-s', '--socket', type=int, default=1, help='Available socket number (Default: 1)')
	argParser.add_argument('--env', choices=['local', 'hdp'], default='local', help='Select running environment (Default: local)')
	argParser.add_argument('--tlc', help='TLC executable path')
	argParser.add_argument('--jobtemp', choices=['ExpressQ', 'Cuda65-Nodes', 'DevNodes'], default='ExpressQ', help='Select job template (only for HDP environment)')

	try:
		args = argParser.parse_args()
	except:
		exit()

	# sanity check
	if not os.path.exists(args.dataset):
		print 'Working directory "%s" doesn\'t exist' % args.dataset
		exit()
	if not os.path.exists(args.logfile):
		print 'Log file "%s" doesn\'t exist' % args.logfile
		exit()

	if args.env == 'local':
		runLocal(args)
	elif args.env == 'hdp':
		runHDP(args.__dict__)
