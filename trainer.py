import os
import sys
import re
import time

from const import Const
from logger import Logger
from task import *
from rsp import RspDescription

THRESHOLD = 0.1

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

def parseLine(line):
	line = line.strip()
	cmd = line[1:line.index(']')]
	content = line[line.index(']')+1:].strip()
	return (cmd, content)

def parseContent(content):
	entry = {}
	p = re.compile('\d+(\.\d+)?')
	for e in content.split():
		data = e.split('=')
		if data[1].startswith('['):
			l = []
			for e in data[1].strip('[]').split(','):
				if p.match(e):
					e = eval(e)
				l.append(e)
			data[1] = l
		elif p.match(data[1]):
			data[1] = eval(data[1])
		entry[data[0]] = data[1]
	return entry

class Trainer(object):
	# NOTE!!: both dataset and logfile should be absolute path
	def __init__(self, dataset, logfile, scheduler):
		# switch to working directory
		os.chdir(dataset)

		self.dataset = dataset
		self.logger = Logger(logfile)
		self.scheduler = scheduler

		self.history = {}
		self.threadName = None
		# rsp file template
		self.rspTmpl = None
		self.epoch = 0
		self.subId = 0 # subId for same epoch
		# learning rate
		self.lr = None
		# learning rate reduction
		self.lred = 0.8
		# NN description for next epoch
		self.nn = None
		# random seed
		self.rs = None
		# dropout
		self.idrop = None # input drop
		self.hdrop = None # hidden drop
		self.dropEpoch = 0 # first epoch to drop (default: 0, drop from beginning)

		self.accuracy = 0.0
		self.meanErr = 1000.0

		self.bestAccuracy = 0.0
		self.bestMeanErr = 1000.0
		self.bestModel = None

		# parse to the end of log
		self.parseLog()
		# recover from any crash
		self.recover()

	def summary(self):
		s = 'Dataset = %s, Thread Name = %s\n' % (os.path.basename(self.dataset), self.threadName)
		s += '\nHistory:\n'
		s += '\n'.join(str(x) for x in self.history.values())
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
		s += '  MeanErr = %s\n' % self.meanErr
		s += '  Accuracy = %s\n' % self.accuracy
		return s
	__str__ = summary

	def parseLog(self):
		# load logfile
		header, logs = self.logger.readToEnd()

		for item in header.split():
			data = item.split('=')
			data[0] = data[0].lower()
			if data[0] == 'name':
				self.threadName = data[1]
			elif data[0] == 'rsptemplate':
				# load template rsp file
				rspTmplFile = data[1]
			elif data[0] == 'lr':
				self.lr = eval(data[1])
			elif data[0] == 'nn':
				self.nn = data[1]
			elif data[0] == 'lred':
				self.lred = eval(data[1])
			elif data[0] == 'rs':
				self.rs = Const.parseValueOrArray(data[1])
			elif data[0] == 'idrop':
				self.idrop = eval(data[1])
			elif data[0] == 'hdrop':
				self.hdrop = eval(data[1])
			elif data[0] == 'dropepoch':
				self.dropEpoch = eval(data[1])
		self.rspTmpl = RspDescription(rspTmplFile, self.idrop, self.hdrop, self.dropEpoch)

		index = 0
		while index < len(logs):
			cmd, content = parseLine(logs[index])
			index += 1

			if cmd == 'JOB NEW':
				job = parseContent(content)
				job['Fork'] = False
				job['Finish'] = False
				self.history[job['Epoch']] = job
				# force LR equal to the history
				self.lr = job['LR']

			elif cmd == 'JOB END':
				entry = parseContent(content)
				entry['Finish'] = True
				job = self.history[entry['Epoch']]
				job.update(entry)
				# update trainer status for next epoch
				self.update(job['MeanErr'], job['Accuracy'], job['Model'], job['Fork'], log=False)

			elif cmd == 'FORKJOB NEW':
				job = parseContent(content)
				job['Fork'] = True
				job['TaskList'] = {}
				job['Finish'] = False
				self.history[job['Epoch']] = job

			elif cmd == 'FORKJOB END':
				entry = parseContent(content)
				entry['Finish'] = True
				job = self.history[entry['Epoch']]
				job.update(entry)
				# update trainer status for next epoch
				self.rs = job['RS']
				self.update(job['MeanErr'], job['Accuracy'], job['Model'], job['Fork'], log=False)

			elif cmd == 'TASK NEW':
				task = parseContent(content)
				task['Finish'] = False
				self.history[self.epoch]['TaskList'][task['SubId']] = task

			elif cmd == 'TASK END':
				entry = parseContent(content)
				entry['Finish'] = True
				self.history[self.epoch]['TaskList'][entry['SubId']].update(entry)

			elif cmd == 'UPDATE':
				data = content.split('=')
				data[0] = data[0].lower()
				if data[0] == 'lr':
					self.lr = eval(data[1])
				elif data[0] == 'lred':
					self.lred = eval(data[1])
				elif data[0] == 'rs':
					self.rs = eval(data[1])

	def recover(self):
		if len(self.history) == 0:
			return
		if self.epoch not in self.history:
			# already advance to next epoch
			return
		if self.history[self.epoch]['Finish']:
			# last epoch finished
			return
		# now start to recover last epoch
		# if job is finished but not written in the log 
		job = self.history[self.epoch]
		if job['Fork']:
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
				self.logger.log('[FORKJOB END] Epoch=%s MeanErr=%s Accuracy=%s Model=%s RS=%s' % (self.epoch, meanErr, accuracy, model, rs))
				# update trainer status for next epoch
				self.rs = rs
				self.logger.log('[UPDATE] RS=%s' % self.rs)
				self.update(meanErr, accuracy, model, job['Fork'], log=True)
		else:
			if os.path.exists(job['Model']) and os.path.exists(job['Stdout']):
				(meanErr, accuracy) = parseTLCResult(job['Stdout'])
				if meanErr is not None and accuracy is not None:
					# job is indeed finished, need to update the log
					self.logger.log('[JOB END] Epoch=%s MeanErr=%s Accuracy=%s' % (self.epoch, meanErr, accuracy))
					job['MeanErr'] = meanErr
					job['Accuracy'] = accuracy
					job['Finish'] = True
					# update trainer status for next epoch
					self.update(meanErr, accuracy, job['Model'], job['Fork'], log=True)
	
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
				task = ForkTask(self.rspTmpl, self.dataset, self.threadName, self.epoch, self.lr, self.nn, self.rs)
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
				task = Task(self.rspTmpl, self.dataset, self.threadName, self.epoch, self.lr, self.nn, self.rs, self.subId)
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

	def update(self, meanErr, acc, model, fork, log=True):
		reduceLR = False
		# decide whether to reduce LR
		if acc < self.accuracy:
			reduceLR = True
		elif acc == self.accuracy:
			if meanErr >= self.meanErr:
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
			if log:
				self.logger.log('[UPDATE] RS=%s' % self.rs)
		else:
			self.epoch += 1
			self.subId = 0
			self.nn = model
			self.accuracy = acc
			self.meanErr = meanErr

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
				if log:
					self.logger.log('[UPDATE] LR=%s' % self.lr)
	