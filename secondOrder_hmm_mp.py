#!/usr/bin/env python -*- coding: utf-8 -*-

import sys
import ctypes
import logging
import warnings
import numpy as np
from time import time
from itertools import izip_longest
from multiprocessing import Process, Array, cpu_count, current_process, Queue
warnings.filterwarnings("ignore")

"""Implementation of Unsupervised Second-order Hidden Morkov Model using multiprocessing."""

__Author__ = "Riyaz Ahmad Bhat"
__Version__ = "1.0"

	
class unsupervisedHMM(object):
	
	def __init__(self, symbols, priors, outputs, transitions_1HMM, transitions_2HMM):
		self._priors      = priors      
		self._symbols     = symbols     
		self._outputs     = outputs    
		self._transitions_1HMM = transitions_1HMM 
		self._transitions_2HMM = transitions_2HMM 
		self._logprob = Array('d',[0])

	def _forward_probability(self, unlabeled_sequence, alpha=None):
		T  = len(unlabeled_sequence)
		N  = self._priors.shape[0]
		O1 = self._symbols[unlabeled_sequence[0]] # NOTE from second observation for second order HMM
		O2 = self._symbols[unlabeled_sequence[1]] # NOTE from second observation for second order HMM

		#NOTE : Update alpha from second observation
		alpha[0] = self._priors + self._outputs[:, O1]
		alpha[1] = self._priors[:,np.newaxis] + self._transitions_1HMM +\
			 self._outputs[:, O1, None] + self._outputs[:, O2]
		# Recursive computation for 2 <= t <= T
		for t in range(2, T):
			symbol = self._symbols[unlabeled_sequence[t]]
			for j in range(self._priors.shape[0]):
				for k in range(self._priors.shape[0]):
					alpha[t, j, k] = self.logsumexp(alpha[t-1][:,j] + self._transitions_2HMM[:, j,k]) 
				alpha[t, j] += self._outputs[:,symbol]

	def _backward_probability(self, unlabeled_sequence, beta=None):
		T = len(unlabeled_sequence)
		N = self._priors.shape[0]
		# Initialise the backward probabilities beta(i,j) at time T
		beta[T-1] = 0.0 #NOTE last observation/word of the sentence
		
		# Recursive computation for T-1 => t => 2
		for t in range(T-2, -1, -1): # NOTE from penultimate word to the second word of the sentence
			symbol = self._symbols[unlabeled_sequence[t+1]] # previous observation in reverse order
			for i in range(self._priors.shape[0]):
				for j in range(self._priors.shape[0]):
					beta[t, i, j] = self.logsumexp(self._transitions_2HMM[i,j] + \
								self._outputs[:,symbol] + beta[t + 1, j])

	def train(self, unlabeled_sequences):
		N = self._priors.shape[0] #number of tags
		M = self._outputs.shape[1] #size of vocabulary

		epsilon        = 1e-6
		iteration      = 0
		converged      = False
		last_logprob   = None
		max_iterations = 50

		sharedArrays = dict()
		workers = cpu_count()
		chunkSize = len(unlabeled_sequences) / workers

		for worker in range(workers):
			sharedArrays.setdefault(worker, {})
			sharedArrays[worker]["eta"] = np.ctypeslib.as_array(\
							Array(ctypes.c_double, N*N*N).get_obj()).reshape(N,N,N)
			sharedArrays[worker]["eta_t1"] = np.ctypeslib.as_array(\
							Array(ctypes.c_double, N*N*N).get_obj()).reshape(N,N,N)
			sharedArrays[worker]["gamma"] = np.ctypeslib.as_array(\
							Array(ctypes.c_double, N*M).get_obj()).reshape(N,M)

		# iterate until convergence
		while not converged and iteration < max_iterations:
			loop_start = time()
			self._logprob[0] = 0
			_eta_    = np.ones((N,N,N)) + float('-1e300')
			_eta_t1_ = np.ones((N,N,N)) + float('-1e300')
			_gamma_  = np.ones((N,M))   + float('-1e300')

			#NOTE: Itererate over unlabelled training instances.
			sequences = izip_longest(*[iter(unlabeled_sequences)] * chunkSize)
			processes = list()	
			for worker in range(workers):
				sharedArrays[worker]["eta"]    += float('-1e300')
				sharedArrays[worker]["eta_t1"] += float('-1e300')
				sharedArrays[worker]["gamma"]  += float('-1e300')

				task = Process(target=self.baum_welch, args=(sequences.next(),
							sharedArrays[worker]["eta"], 
								sharedArrays[worker]["eta_t1"], 
									sharedArrays[worker]["gamma"],))
				task.start()
				processes.append(task)
			for  p in processes:p.join()
			for worker in range(workers):
				_eta_[:]   = self.logsumexp(np.array([_eta_, sharedArrays[worker]["eta"]]), axis=0)
				_eta_t1_[:] = self.logsumexp(np.array([_eta_t1_, sharedArrays[worker]["eta_t1"]]), axis=0)
				_gamma_[:] = self.logsumexp(np.array([_gamma_, sharedArrays[worker]["gamma"]]), axis=0)

			# use the calculated values to update the transition and output probability values
			_gamma_i = self.logsum(_gamma_, axis=1)[:,np.newaxis]
			self._transitions_2HMM = _eta_ - self.logsumexp(_eta_.reshape(N*N, N), axis=1).reshape(N,N, 1)
			_xi_ = self.logsumexp(_eta_t1_.reshape(N*N, N), axis=1).reshape(N,N)
			self._transitions_1HMM = _xi_ - self.logsumexp(_xi_, axis=1)[:,np.newaxis]

			temp_emission = _gamma_ - _gamma_i
			indices = temp_emission != float('-1e300')
			self._outputs[indices] = temp_emission[indices]

			loop_end = time()	
			if iteration > 0 and abs(self._logprob[0] - last_logprob) < epsilon:
				converged = True
			print >> sys.stderr, "Time taken from iteration %d to iteration %d is %f, likelihood=%f" \
				% (iteration, iteration+1, (loop_end - loop_start) / 60, self._logprob[0])

			iteration += 1
			last_logprob = self._logprob[0]
			
			np.save("bigram-para-"+str(iteration+1), self._transitions_1HMM)
			np.save("trigram-para-"+str(iteration+1), self._transitions_2HMM)
			np.save("emission-para-"+str(iteration+1), self._outputs)
		return self
		
	
	def baum_welch(self,sequences, _eta_, _eta_t1_, _gamma_=None):
		N = self._priors.shape[0] #number of tags
		M = self._outputs.shape[1] #size of vocabulary

		for sequence in sequences:
			sequence = sequence.split()
			T = len(sequence)
			# compute forward and backward probabilities
			alpha = np.ctypeslib.as_array(Array(ctypes.c_double, T*N*N).get_obj()).reshape(T,N,N)
			beta = np.ctypeslib.as_array(Array(ctypes.c_double, T*N*N).get_obj()).reshape(T,N,N)

			p=[Process(target=self._forward_probability, args=(sequence,alpha,)), 
				Process(target=self._backward_probability, args=(sequence,beta,))]
			for pi in p:pi.start()
			for pj in p:pj.join()
						
			lpk =	self.logsum(alpha[T-1].ravel()) # last words alpha
			self._logprob[0] += lpk
			alpha_FO = self.logsumexp(alpha, axis=1) # should be colom sum i.e axis=1
			beta_FO = self.logsumexp(beta, axis=1) # should be colom sum i.e axis=1
			
			eta_t1_temp = np.zeros((N,N,N))
			for t in xrange(T):
				x_idx = self._symbols[sequence[t]]
				if t < T-2:
					xnext = self._symbols[sequence[t+2]]
					eta_temp = np.zeros((N,N,N))

				for i in range(N):
					for j in range(N):
						if t < T-2:
							eta_ijk = (alpha[t+1,i,j] + self._transitions_2HMM[i,j] + \
                                                                                self._outputs[:,xnext] + beta[t+2, j]) - lpk
							eta_temp[i,j] = eta_ijk
							if t==0:eta_t1_temp[i,j] = eta_ijk
				jp_n = alpha_FO[t] + beta_FO[t] - lpk
				_gamma_[:, x_idx] = self.logsumexp(np.array([_gamma_[:,x_idx],jp_n]), axis=0)
				if t < T-2:
					_eta_[:] = self.logsumexp(np.array([_eta_, eta_temp]), axis=0)
			_eta_t1_[:] = self.logsumexp(np.array([_eta_t1_, eta_t1_temp]), axis=0)

	def logsumexp(self, a, axis=None):
		if axis:
			a = np.rollaxis(a, axis)
		a_max = a.max(axis=0)
		out = np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max
		return out

	def logsum(self, a, axis=None):
		a = np.asarray(a)
		if axis==1:
			a = a.T
		a_max = a.max(axis=0)
		out = np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max
		return out
