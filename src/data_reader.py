import gzip
import json
import numpy as np
#import simplejson as json
import zipfile

class Embeddings:	
	# Get embedding here:
	# http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc

	def __init__(self):
		self.START_token = 0
		self.END_token = 1
		self.UNK_token = None
		self.token2index = {'<start>': 0, '<end>': 1}
		self.index2token = {0: '<start>', 1: '<end>'}		
		self.data = None

	@staticmethod
	def load(embeddings_filepath, tokens_filepath):
		embeddings = Embeddings()

		# Filter tokens
		valid_tokens = set()
		with open(tokens_filepath, encoding='utf-8') as fp:
			for token in fp:
				valid_tokens.add(token.strip())

		# Get file
		z = zipfile.ZipFile(embeddings_filepath, 'r')
		namelist = z.namelist()
		if len(namelist) != 1:
			return None
		filename =  namelist[0]

		# Memory to allocate
		# Read line-by-line due to memory contraints
		num_embeddings = 0
		for row, line in enumerate(z.open(filename)):
			line = line.decode('utf-8')
			if row == 0: # HEADER
				_, embedding_size = line.split(' ')
				embedding_size = int(embedding_size)
			else:
				token, embedding_str = line.split(' ', 1)
				if token in valid_tokens:
					num_embeddings+=1

		embeddings.data = np.zeros((num_embeddings+2, embedding_size), dtype=np.float32)
		embeddings.data[0] = 0.05 * np.random.randn(embedding_size)
		embeddings.data[1] = 0.05 * np.random.randn(embedding_size)

		# Read line-by-line due to memory contraints
		for row, line in enumerate(z.open(filename)):
			if row == 0: # HEADER
				continue

			line = line.decode('utf-8')
			token, embedding_str = line.split(' ', 1)
			if token not in valid_tokens:
				continue

			index = len(embeddings.token2index)
			embeddings.token2index[token] = index
			embeddings.index2token[index] = token
			embeddings.data[index] = np.fromstring(embedding_str, dtype=np.float32, sep=' ')
		z.close()

		if '<unk>' in embeddings.token2index:
			embeddings.UNK_token = embeddings.token2index['<unk>']

		return embeddings

	def __getitem__(self, index):
		return self.data[index]

	def words_to_indices(self, text):
		if self.UNK_token:
			return [self.token2index.get(token, self.UNK_token) for token in text.split(' ')]
		else:
			return [self.token2index[token] for token in text.split(' ') if token in token2index]

class ResumeDataset:

	def __init__(self):
		self.data = []

	@staticmethod
	def load(resumes_filepath):
		# Store resume string due to memory contraints
		dataset = ResumeDataset()
		with gzip.open(resumes_filepath, mode='rt', encoding='utf-8') as z:
			for i, resume_str in enumerate(z):
				dataset.data.append(resume_str)
		return dataset

	def __getitem__(self, index):
		return json.loads(self.data[index])