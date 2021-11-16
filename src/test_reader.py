from data_reader import Embeddings
from data_reader import ResumeDataset

data_dirpath = '../data'
embeddings_filepath = data_dirpath + '/glove_s100.zip'
tokens_filepath = data_dirpath + '/resumes_we2_edu1_about100_[tokens].txt'
resumes_filepath = data_dirpath + '/resumes_we2_edu1_about100_[preprocessed].txt.zip'


print('\n# Load embeddings...')
embeddings = Embeddings.load(embeddings_filepath, tokens_filepath)
print('\nShape:\n{}'.format(embeddings.data.shape))
print('\nEmbeddings[-1]:\n{}'.format(embeddings[-1]))

print('\n# Load resumes...')
resumes = ResumeDataset.load(resumes_filepath)
resume_data = resumes[3]
highlights = resume_data['highlights']

print('\nResumes[3][\'highlights\'] (raw):\n{}'.format(highlights))
print('\nResumes[3][\'highlights\'] (indices):\n{}'.format(embeddings.words_to_indices(highlights)))