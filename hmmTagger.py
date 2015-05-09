#!/usr/bin/env python -*- coding: utf-8 -*- 

import sys	
import argparse
import numpy as np

__Author__ = "Riyaz Ahmad Bhat"
__Title__ = "Second-order HMM tagger for Indian languages."

class Tagger (object):

	def __init__ (self, symbols, priors, bigrams, trigrams, emissions, states=None):
		self._states      = states
		self._priors      = priors        # NOTE a numpy array
		self._symbols     = symbols       # NOTE a dictionary of symbols and their indices
		self._outputs     = emissions     # NOTE a numpy array
		self._transitions_1HMM = bigrams  # NOTE a numpy array
		self._transitions_2HMM = trigrams # NOTE a numpy array

	def HMM_2_Decoding(self, unlabelled_sequence):
		T  = len(unlabelled_sequence)
		N  = self._priors.shape[0]
		O1 = self._symbols[unlabelled_sequence[0]] # NOTE from second observation for second order HMM
		O2 = self._symbols[unlabelled_sequence[1]] # NOTE from second observation for second order HMM
	
		viterbi = np.zeros((T,N,N))
		#NOTE Initialize for second observation
		'''
		initialStateProbabilities = self._priors[:,np.newaxis] + self._outputs[:, O1, None]
		for l in range(N):
			for m in range(N):
				temp = list()
				for i in range(N):
					temp.append(initialStateProbabilities[i] + self._transitions_1HMM[l,m])
				viterbi[1, l,m] = np.max(np.array(temp) + self._outputs[m, O2])
		'''

		viterbi[1] = self._priors[:,np.newaxis] + self._outputs[:, O1, None] +\
					 self._transitions_1HMM + self._outputs[:, O2]
		backtracking = np.zeros((T,N,N), int)

		# Recursive computation for 2 <= t <= T
		for t in range(2, T):
			symbol = self._symbols[unlabelled_sequence[t]]
			for j in range(self._priors.shape[0]):
				for k in range(self._priors.shape[0]):
					temp = list()
					for i in range(self._priors.shape[0]):
						temp.append(viterbi[t-1, i, j] + \
								self._transitions_2HMM[i][j][k])
					viterbi[t, j,k] = np.max(np.array(temp) + self._outputs[k][symbol])
					backtracking[t, j, k] = np.argmax(temp)
		
		PenUltimateState, UltimateState = np.unravel_index(np.argmax(viterbi[-1]),(N,N))
		best_tag_sequence = [UltimateState, PenUltimateState]
		for t in range(T-3, -1, -1):#NOTE Backtrack
			best_tag_sequence.append(backtracking[t+2, best_tag_sequence[-1], best_tag_sequence[-2]])
		best_tag_sequence.reverse()
		return best_tag_sequence

	def HMM_1_Decoding(self, unlabelled_sequence):
		T  = len(unlabelled_sequence)
		N  = self._priors.shape[0]

		viterbi = np.zeros((T,N))
		backtracking = np.zeros((T,N), int) * -1
		
		viterbi[0] = self._priors + self._outputs[:, self._symbols[unlabelled_sequence[0]]]
		for t in range(1, T):
			symbol = self._symbols[unlabelled_sequence[t]]
			for j in range(N):
				temp = list()
				for i in range(N):
					temp.append(viterbi[t-1, i] + self._transitions_1HMM[i,j])
				viterbi[t, j] = np.max(temp) + self._outputs[j, symbol]
				backtracking[t, j] = np.argmax(temp)

		UltimateState = np.argmax(viterbi[-1])
		best_tag_sequence = [UltimateState]
		#print backtracking
		for t in range(T-2, -1, -1):
			best_tag_sequence.append(backtracking[t+1,best_tag_sequence[-1]])
		best_tag_sequence.reverse()
		return best_tag_sequence

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Second-Order HMM Tagger.")
	parser.add_argument('--input-file'              , dest='input'     , required=True, help='Unlabelled sentences')
	parser.add_argument('--states-dictionary'       , dest='states'    , required=True, help='Dictionary of states')
	parser.add_argument('--symbols-dictionary'      , dest='symbols'   , required=True, help='Dictionary of Vocabulary')
	parser.add_argument('--prior-probabilities'     , dest='prior'     , required=True, help='Array of prior probabilities')
	parser.add_argument('--emission-probabilities'  , dest='emission'  , required=True, help='Array of emission probabilities')
	parser.add_argument('--bigram-transitions'      , dest='bigram'    , required=True, help='Array of transition_probabilities')
	parser.add_argument('--trigram-transitions'     , dest='trigram'   , required=True, help='Array of transition_probabilities')

	args = parser.parse_args()		

	prior_probabilities = np.load(args.prior)
	emission_probabilities = np.load(args.emission)
	transition_probabilities_1HMM = np.load(args.bigram)
	transition_probabilities_2HMM = np.load(args.trigram)
	state_symbols = np.load(args.states)[0]
	observation_symbols = np.load(args.symbols)[0]

	unlabelled_sequences = open(args.input).read().split("\n\n")

	trainingSentences = set()
	for sequence in unlabelled_sequences:
		instances = sequence.decode("utf-8").split("\n")
		if len(instances) < 5:continue
		for instance in instances:
			word, tag = instance.split("\t")
			if word not in observation_symbols:
				break
		else:
			trainingSentences.add(sequence.strip().decode("utf-8"))
	model = Tagger(observation_symbols, prior_probabilities, transition_probabilities_1HMM, 
			transition_probabilities_2HMM, emission_probabilities, state_symbols)

	print >> sys.stderr, len(trainingSentences)
	for sentence in trainingSentences:
		instances = sentence.split("\n")
		words = [i.split("\t")[0] for i in instances]
		goldTags = [i.split("\t")[1] for i in instances]
		autoTags = model.HMM_2_Decoding(words)
		#tags = model.viterbi(words)
		tagSequence = list()
		for tag in autoTags:
			for key, value in state_symbols.items():
				if tag == value:tagSequence.append(key)
		print "\n".join([word.encode("utf-8")+"\t"+goldTag.encode("utf-8")+"\t"+autoTag.encode("utf-8")\
				for word, goldTag, autoTag in zip(words, goldTags, tagSequence)])
		print
