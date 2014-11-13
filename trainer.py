import os
import sys
import re
import time
import subprocess

from const import Const
from logger import Logger
from task import *
from rsp import *
from nn import NeuralNetwork

THRESHOLD = 0.1
DEFAULT_LRED = 0.8

def parseTLCResult(fn):
	# loss function reduction
	meanErr = None
	# Accuracy on test data
	accuracy = None
	with open(fn) as f:
		state = 0
		for line in f:
			if line.startswith('Iter'):
				for token in line.split(', '):
					if 'MeanErr' in token:
						meanErr = eval(token[token.index('=')+1:token.index('(')])
			elif accuracy is None and line.startswith('ACCURACY(micro-avg)'):
				accuracy = eval(line.strip().split(':')[1])
			else:
				continue
	return (meanErr, accuracy)

class Trainer(object):
	def summary(self):
		pass

	def train(self):
		pass

class RegularTrainer(Trainer):
	def __init__(self, logger, dataset, scheduler, rspTmpl=None):
		self.logger = logger
		self.dataset = dataset
		self.scheduler = scheduler

		# load from logger header
		headers = self.logger.headers

		# Constant information
		self.threadName = headers['ThreadName']
		# random seed
		self.rs = headers['rs'] if 'rs' in headers else None
		# learning rate reduction
		self.lred = headers['lred'] if 'lred' in headers else DEFAULT_LRED
		# drop information
		self.idrop = headers['idrop'] if 'idrop' in headers else None
		self.hdrop = headers['hdrop'] if 'hdrop' in headers else None
		self.dropEpoch = headers['DropEpoch'] if 'DropEpoch' in headers else None

		# Create rsp generator
		if rspTmpl is not None:
			# use customized rsp template
			self.rsp = RspGenerator(rspTmpl, self.idrop, self.hdrop, self.dropEpoch)
		else:
			self.rsp = RspGenerator(headers['RspTmplFile'], self.idrop, self.hdrop, self.dropEpoch)
			# Set specific train and test dataset
			if 'TrainDataset' in headers:
				self.rsp.rspTmpl.trainDataset = headers['TrainDataset']
			if 'TestDataset' in headers:
				self.rsp.rspTmpl.testDataset = headers['TestDataset']

		# initialize training status
		self.epoch = 0
		self.subId = 0 # subId for same epoch
		self.lr = headers['lr'] # learning rate
		self.nn = headers['nn'] # NN file for next epoch

		# Accuracy and MeanErr for last epoch
		self.lastAccuracy = 0.0	
		self.lastMeanErr = 1000.0

		# Record best result
		self.bestAccuracy = 0.0
		self.bestMeanErr = 1000.0
		self.bestModel = None

		# recover from logger
		self.recover()

	def summary(self):
		s = '[Regular Trainer]\n'
		s += 'Dataset = %s, Thread Name = %s\n' % (os.path.basename(self.dataset), self.threadName)
		s += '\nHistory:\n'
		s += '\n'.join(str(x) for x in self.logger.history)
		s += '\n\nStatus:\n'
		s += '  Epoch = %s\n' % self.epoch
		if self.subId > 0:
			s += '  SubId = %s\n' % self.subId
		s += '  LR = %s\n' % self.lr
		s += '  LRed = %s\n' % self.lred
		s += '  NN = %s\n' % self.nn
		s += '  RS = %s\n' % self.rs
		s += '  idrop = %s\n' % self.idrop
		s += '  hdrop = %s\n' % self.hdrop
		s += '  DropEpoch = %s\n' % self.dropEpoch
		s += '  MeanErr = %s\n' % self.bestMeanErr
		s += '  Accuracy = %s\n' % self.bestAccuracy
		return s
	__str__ = summary

	# recover traingin status from logger history
	def recover(self):
		history = self.logger.history
		if len(history) == 0:
			return
		# check if last job finishes
		lastJob = history[-1]
		if not lastJob['Finish']:
			self.tryRecoverJob(lastJob)
			if not lastJob['Finish']:
				history = history[:-1] # remove unfinished job
		for job in history:
			self.update(job['MeanErr'], job['Accuracy'], job['Model'], job['Fork'])
			if job['Fork']:
				self.rs = job['RS']

	# check if job finished but not written in the log
	def tryRecoverJob(self, job):
		if job['Fork']:
			# recover fork job
			taskList = job['TaskList']
			if len(taskList) == 0:
				# job hasn't started
				return
			# start to recover
			meanErr = 1000
			accuracy = 0
			model = None
			rs = None
			for task in taskList.values():
				if not task['Finish']:
					# recover single task
					if os.path.exists(task['Model']) and os.path.exists(task['Stdout']):
						(err, acc) = parseTLCResult(task['Stdout'])
						if err is not None and acc is not None:
							task['MeanErr'] = err
							task['Accuracy'] = acc
							task['Finish'] = True
							self.logger.log('[TASK END] SubId=%s MeanErr=%s Accuracy=%s' % (task['SubId'], err, acc))
				if task['Finish']:
					err = task['MeanErr']
					acc = task['Accuracy']
					if acc > accuracy or (acc == accuracy and err < meanErr):
						accuracy = acc
						meanErr = err
						model = task['Model']
						rs = task['RS']
			if model is not None:
				job['MeanErr'] = meanErr
				job['Accuracy'] = accuracy
				job['Model'] = model
				job['RS'] = rs
				job['Finish'] = True
				self.logger.log('[FORKJOB END] Epoch=%s MeanErr=%s Accuracy=%s Model=%s RS=%s' % (job['Epoch'], meanErr, accuracy, model, rs))
		else:
			if os.path.exists(job['Model']) and os.path.exists(job['Stdout']):
				(meanErr, accuracy) = parseTLCResult(job['Stdout'])
				if meanErr is not None and accuracy is not None:
					# job is indeed finished, need to update the log
					job['MeanErr'] = meanErr
					job['Accuracy'] = accuracy
					job['Finish'] = True
					self.logger.log('[JOB END] Epoch=%s MeanErr=%s Accuracy=%s' % (job['Epoch'], meanErr, accuracy))
	
	# update training status
	def update(self, meanErr, acc, model, fork):
		reduceLR = False
		# decide whether to reduce LR
		if acc < self.lastAccuracy:
			reduceLR = True
		elif acc == self.lastAccuracy and meanErr >= self.lastMeanErr:
			reduceLR = True
		
		# decide whether to advance to next epoch
		if self.epoch == 0 and acc < THRESHOLD:
			if fork:
				# all forks are bad, stop training
				print 'All forks have bad results. Stop now.'
				exit()
			# if the accuracy of 1st epoch is too low, then change a random seed
			self.rs = 1 if self.rs is None else self.rs+1
			# keep stay at epoch 0, increase subId
			self.subId += 1
		else:
			self.epoch += 1
			self.subId = 0
			self.nn = model
			self.lastAccuracy = acc
			self.lastMeanErr = meanErr

			if acc > self.bestAccuracy:
				self.bestAccuracy = acc
				self.bestMeanErr = meanErr
				self.bestModel = model
			elif acc == self.bestAccuracy and self.bestMeanErr > meanErr:
				self.bestAccuracy = acc
				self.bestMeanErr = meanErr
				self.bestModel = model
			if reduceLR:
				self.lr *= self.lred

	def train(self):
		print '\nStart training thread %s' % self.threadName

		while True:
			if not self.trainOneEpoch():
				print 'Error in training'
				return False
			if self.lr < 1e-4:
				print 'LR %s is too small. Stop training' % self.lr
				break

		if self.bestModel is not None:
			self.logger.log('[FINISH] MeanErr=%s Accuracy=%s Model=%s' % (self.bestMeanErr, self.bestAccuracy, self.bestModel))
		print '============================================'
		print 'Training finished'
		return True

	def trainOneEpoch(self):
		succ = False
		errcnt = 0
		fork = type(self.rs) is list

		print '============================================'
		print 'Start training epoch %s' % self.epoch

		while errcnt < 3:				
			if fork:
				task = ForkTask(self.rsp, self.dataset, self.threadName, self.epoch, self.lr, self.nn, self.rs)
				if not self.scheduler.execute(task):
					print 'Error in executing task'
					return False
				self.logger.log('[FORKJOB NEW] Epoch=%s RS=[%s]' % (self.epoch, ','.join(str(v) for v in self.rs)))
				for t in task.taskList:
					text = '[TASK NEW] SubId=%s LR=%s Model=%s Stdout=%s' % (t.subId, t.lr, t.textModel, t.stdout)
					if t.rs is not None:
						text += ' RS=%s' % t.rs
					self.logger.log(text)
				print 'ForkTask %s has been submitted' % task.taskName
			else:
				task = Task(self.rsp, self.dataset, self.threadName, self.epoch, self.lr, self.nn, self.rs, self.subId)
				if not self.scheduler.execute(task):
					print 'Error in executing task'
					return False
				text = '[JOB NEW] Epoch=%s LR=%s Model=%s Stdout=%s' % (task.epoch, task.lr, task.textModel, task.stdout)
				if task.rs is not None:
					text += ' RS=%s' % task.rs
				self.logger.log(text)
				print 'Task %s has been submitted' % task.taskName

			# wait for job to finish
			while (True):
				time.sleep(60) # check every 1 minute
				if (self.scheduler.checkFinish(task)):
					break

			# now parse the result
			if fork:
				meanErr = 1000
				accuracy = 0
				model = None
				rs = None

				for t in task.taskList:
					(err, acc) = parseTLCResult(t.stdout)
					if err is not None and acc is not None:
						self.logger.log('[TASK END] SubId=%s MeanErr=%s Accuracy=%s' % (t.subId, err, acc))
						if acc > accuracy or (acc == accuracy and err < meanErr):
							accuracy = acc
							meanErr = err
							model = t.textModel
							rs = t.rs
				if model is not None:
					succ = True
					self.rs = rs # pick the best random seed
					self.logger.log('[FORKJOB END] Epoch=%s MeanErr=%s Accuracy=%s Model=%s RS=%s' % (self.epoch, meanErr, accuracy, model, rs))
					self.logger.log('[UPDATE] RS=%s' % self.rs)
			else:
				meanErr = None
				accuracy = None

				(meanErr, accuracy) = parseTLCResult(task.stdout)
				model = task.textModel
				if meanErr is not None and accuracy is not None:
					succ = True
					self.logger.log('[JOB END] Epoch=%s MeanErr=%s Accuracy=%s' % (self.epoch, meanErr, accuracy))

			if succ:
				print 'Epoch %s finished, MeanErr %s, Accuracy %s' % (self.epoch, meanErr, accuracy)
				self.update(meanErr, accuracy, model, fork)
				break
			else:
				errcnt += 1

		return succ

class SharedTrainer(Trainer):
	def __init__(self, logger, dataset, scheduler):
		self.logger = logger
		self.dataset = dataset		
		self.scheduler = scheduler

		# load from logger header
		headers = self.logger.headers
		assert('TrainedNN' in headers)
		assert('LabelFile' in headers)

		self.threadName = headers['ThreadName']
		self.rspTmpl = RspTemplate.parseRspFile(headers['RspTmplFile'])
		self.sharedLayers = headers['Shared']
		self.trainedNNFile = headers['TrainedNN']

		# bottom and top NN files
		self.bottomNNFile = "%s.bottom.nn" % self.threadName
		self.bottomBinFile = "%s.bottom.bin" % self.threadName
		self.topNNFile = "%s.top.nn" % self.threadName # to be trained

		# train and test dataset files
		self.trainDataset = "%s.train.txt" % self.threadName
		self.trainDatasetBin = "%s.train.txt.tlcbin" % self.threadName
		self.testDataset = "%s.test.txt" % self.threadName
		self.testDatasetBin = "%s.test.txt.tlcbin" % self.threadName

		# load label mapping file
		# mapping from original label to new label
		self.labelFile = headers['LabelFile']
		self.labelMap = {}
		self.dimOutput = 0
		self.loadLabelMap()

		# generate bottomNN and topNN
		self.generateNN()
		# update initial nn file in logger's headers
		self.logger.headers['nn'] = self.topNNFile
		# return

		# generate train and test dataset
		self.generateDataset('train')
		self.generateDataset('test')
		# change train and test dataset in rsp template
		self.rspTmpl.trainDataset = self.trainDatasetBin
		self.rspTmpl.testDataset = self.testDatasetBin
		self.rspTmpl.options['/inst'] = 'BinaryInstances'
		if '/instset' in self.rspTmpl.options:
			# no input scale
			del self.rspTmpl.options['/instset']

		self.trainer = RegularTrainer(logger, dataset, scheduler, self.rspTmpl)

	def loadLabelMap(self):
		with open(self.labelFile) as f:
			for line in f:
				items = line.strip().split(',')
				self.labelMap[items[0].strip()] = items[1].strip()
		self.dimOutput = len(set(self.labelMap.values()))

	def generateNN(self):
		# parse trained NN
		fin = open(self.trainedNNFile)
		nn = NeuralNetwork.parseNN(fin)
		# bottomNN are shared, topNN is training target
		bottomNN, topNN = nn.split(self.sharedLayers)
		# change the output size of topNN
		topNN.layers[-1].dimOutput = self.dimOutput

		if not os.path.exists(self.topNNFile):
			print "[SHARED] Creating top nn file"
			with open(self.topNNFile, 'w') as fout:
				fout.write(str(topNN))

		if not os.path.exists(self.bottomNNFile):
			# first write bottomNN in text format
			print "[SHARED] Creating bottom model in text format"
			begin = time.time()
			with open(self.bottomNNFile, 'w') as fout:
				fout.write("%s\n" % str(bottomNN))
				Const.output(fout, fin)
			end = time.time()
			print "[SHARED] Finish in %.1f s" % (end-begin)
		fin.close() # no longer need it

		if not os.path.exists(self.bottomBinFile):
			# generate Rsp that converts text format nn to binary format
			print "[SHARED] Creating bottom model in binary format"
			begin = time.time()
			genBottomRsp = "%s.bottom.gen.rsp" % self.threadName
			with open(genBottomRsp, 'w') as fout:
				fout.write("/c Train %s\n" % self.rspTmpl.testDataset) # usually test dataset is smaller
				if '/inst' in self.rspTmpl.options:
					fout.write("/inst %s\n" % self.rspTmpl.options['/inst'])
				fout.write("/cl mcnn { filename=%s iter=0 }\n" % self.bottomNNFile)
				fout.write("/m %s" % self.bottomBinFile)
				fout.write("/cacheinst-\n")
				fout.write("/threads-\n")

			# run TLC to generate binary model
			command = "%s @%s" % (self.scheduler.tlcpath, genBottomRsp)
			proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
			proc.communicate()
			end = time.time()

			if not os.path.exists(self.bottomBinFile):
				raise Exception("Error in generating bottom model in binary format")

			print "[SHARED] Finish in %.1f s" % (end-begin)

	# datasetType: train,test
	def generateDataset(self, datasetType):
		originDataset = self.rspTmpl.__dict__["%sDataset" % datasetType]
		tmpDataset = "%s.%s.tmp.txt" % (self.threadName, datasetType)
		dataset = self.__dict__["%sDataset" % datasetType]
		datasetBin = self.__dict__["%sDatasetBin" % datasetType]

		if not os.path.exists(dataset):
			# pass the bottom model to dataset
			print "[SHARED] Generating %s tmp dataset" % datasetType
			begin = time.time()
			genRsp = "%s.%s.gen.rsp" % (self.threadName, datasetType)
			with open(genRsp, 'w') as fout:
				fout.write("/c Test %s\n" % originDataset)
				if '/inst' in self.rspTmpl.options:
					fout.write("/inst %s\n" % self.rspTmpl.options['/inst'])
				if '/instset' in self.rspTmpl.options:
					fout.write("/instset %s\n" % self.rspTmpl.options['/instset'])
				fout.write("/m %s\n" % self.bottomBinFile)
				fout.write("/raw %s" % tmpDataset)
				fout.write("/cacheinst-\n")
				fout.write("/threads-\n")
			# run TLC to generate binary model
			command = "%s @%s" % (self.scheduler.tlcpath, genRsp)
			proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
			proc.communicate()
			end = time.time()
			if not os.path.exists(tmpDataset):
				raise Exception("Error in generating %s dataset" % datasetType)
			print "[SHARED] Finish in %.1f s" % (end-begin)

			# now update to the new label
			print "[SHARED] Replacing labels"
			begin = time.time()
			fout = open(dataset, 'w')
			with open(tmpDataset) as fin:
				for line in fin:
					data = line.strip().split("\t")
					data[0] = self.labelMap[data[0]]
					fout.write("%s\n" % "\t".join(data))
			fout.close()
			end = time.time()
			print "[SHARED] Finish in %.1f s" % (end-begin)
			os.remove(tmpDataset)

		if not os.path.exists(datasetBin):
			# convert dataset to binary format
			print "[SHARED] Converting %s dataset to binary format" % datasetType
			begin = time.time()
			makebinRsp = "%s.%s.MakeBin.rsp" % (self.threadName, datasetType)
			with open(makebinRsp, 'w') as fout:
				fout.write("/c CreateInstances %s\n" % dataset)
				fout.write("/instancesClass TextInstances\n")
				fout.write("/instanceWriter BinaryInstanceWriter { ltype=u2 ftype=r4 dense=+ }")
				fout.write("/cacheinst-\n")
				fout.write("/threads-\n")
			# run TLC to generate binary model
			command = "%s @%s" % (self.scheduler.tlcpath, makebinRsp)
			proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
			proc.communicate()
			end = time.time()
			if not os.path.exists(datasetBin):
				raise Exception("Error in making binary for %s dataset" % datasetType)
			print "[SHARED] Finish in %.1f s" % (end-begin)

	def summary(self):
		s = "[Shared Trainer]\n"
		s += "Shared Layers = %s\n" % self.sharedLayers
		s += "Bottom Model = %s\n" % self.bottomBinFile
		s += "Top NN = %s\n" % self.topNNFile
		s += "Train Dataset = %s\n" % self.trainDatasetBin
		s += "Test Dataset = %s\n" % self.testDatasetBin
		s += "\n%s" % self.trainer.summary()
		return s

	def train(self):
		self.trainer.train()

