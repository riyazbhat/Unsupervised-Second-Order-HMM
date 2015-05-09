#!/usr/bin/env python

import sys
import json
import random
import tempfile
import argparse
import numpy as np
import secondOrder_hmm_mp as UNHMM
from multiprocessing import cpu_count

def _train (data, symbols, priors, outputs, transitions_1HMM, transitions_2HMM):
	model = UNHMM.unsupervisedHMM(symbols, priors, outputs, transitions_1HMM, transitions_2HMM)
	model.train(data)
	return model
    
if __name__ == "__main__":


	parser = argparse.ArgumentParser(description="Un-supervised HMM tagger")
	parser.add_argument('--input-file'              , dest='input'      , required=True, help='Un-labelled corpora')
	parser.add_argument('--prior-probabilities'     , dest='prior'      , required=True, help='List of prior probabilities')
	parser.add_argument('--emission-probabilities'  , dest='emission'   , required=True, help='Dict of emission probabilities')
	parser.add_argument('--bigram-transitions'      , dest='bigram'    , required=True, help='Dict of transition_probabilities')
	parser.add_argument('--trigram-transitions'     , dest='trigram'   , required=True, help='Dict of transition_probabilities')
	parser.add_argument('--symbols-dictionary'      , dest='symbols'   , required=True, help='Dict of transition_probabilities')

	args = parser.parse_args()

	prior_probabilities = np.load(args.prior)
	emission_probabilities = np.load(args.emission)
	transition_probabilities_1HMM = np.load(args.bigram)
	transition_probabilities_2HMM = np.load(args.trigram)
	observation_symbols = np.load(args.symbols)[0]

	unlabelled_sequences = open(args.input)
	trainingSentences = set()
	for sequence in unlabelled_sequences:
		words = sequence.decode("utf-8").split()
		if len(words) < 5:continue
		for word in words:
			if word not in observation_symbols:
				break
		else:
			trainingSentences.add(sequence.strip().decode("utf-8"))
	
	unlabelled_sequences.close()
	skip = len(trainingSentences) % cpu_count()	
	if skip:
		trainingSentences = list(trainingSentences)[:-skip]
	print len(trainingSentences)
	model = _train(trainingSentences, observation_symbols, prior_probabilities, emission_probabilities,
						transition_probabilities_1HMM, transition_probabilities_2HMM)
	np.save("emission_parameters.npy" , model._outputs)
	np.save("bigram_parameters.npy"   , model._transitions_1HMM)
	np.save("trigram_parameters.npy"  , model._transitions_2HMM)
