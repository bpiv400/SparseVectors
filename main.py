import os
import csv
import subprocess
import re
import random
import numpy as np
from math import log, sqrt, exp


def read_in_shakespeare():
  '''Reads in the Shakespeare dataset processesit into a list of tuples.
     Also reads in the vocab and play name lists from files.

  Each tuple consists of
  tuple[0]: The name of the play
  tuple[1] A line from the play as a list of tokenized words.

  Returns:
    tuples: A list of tuples in the above format.
    document_names: A list of the plays present in the corpus.
    vocab: A list of all tokens in the vocabulary.
  '''

  tuples = []

  with open('will_play_text.csv') as f:
    csv_reader = csv.reader(f, delimiter=';')
    for row in csv_reader:
      play_name = row[1]
      line = row[5]
      line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
      line_tokens = [token.lower() for token in line_tokens]

      tuples.append((play_name, line_tokens))

  with open('vocab.txt') as f:
    vocab =  [line.strip() for line in f]

  with open('play_names.txt') as f:
    document_names =  [line.strip() for line in f]

  return tuples, document_names, vocab

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
  '''Returns a numpy array containing the term document matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary

  Let m = len(vocab) and n = len(document_names).

  Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of words
        and each column corresponds to a document. A_ij contains the
        frequency with which word i occurs in document j.
  '''
  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  docname_to_id = dict(zip(document_names, range(0, len(document_names))))

  # YOUR CODE HERE
  matrix = np.zeros((len(vocab), len(document_names)))

  for play, line in line_tuples:
    for word in line:
      matrix[(vocab_to_id[word], docname_to_id[play])] += 1

  return matrix

def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
  '''Returns a numpy array containing the term context matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    vocab: A list of the tokens in the vocabulary

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

  Let n = len(vocab).

  Returns:
    tc_matrix: A nxn numpy array where A_ij contains the frequency with which
        word j was found within context_window_size to the left or right of
        word i in any sentence in the tuples.
  '''
  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

  matrix = np.zeros((len(vocab), len(vocab)))
  for play, line in line_tuples:
    for i in range(0, len(line)):
      min_context = i - context_window_size
      if min_context < 0:
        min_context = 0
      max_context = i + context_window_size
      if max_context >= len(line):
        max_context = len(line) - 1

      for j in range(min_context, max_context + 1):
        if j != i:
          matrix[vocab_to_id[line[i]], vocab_to_id[line[j]]] += 1  
  matrix = matrix + pow(10, -3)
  return matrix

def create_PPMI_matrix(term_context_matrix):
  '''Given a term context matrix, output a PPMI matrix.
  
  See section 15.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.
  
  Input:
    term_context_matrix: A nxn numpy array, where n is
        the numer of tokens in the vocab.
  
  Returns: A nxn numpy matrix, where A_ij is equal to the
     point-wise mutual information between the ith word
     and the jth word in the term_context_matrix.
  '''
  def divide_array(matrix_array, marginal_array):
    return matrix_array - marginal_array

  print(str(term_context_matrix))
  context_sum = np.sum(term_context_matrix, axis=0)
  word_sum = np.sum(term_context_matrix, axis=1)

  total_sum1 = np.sum(context_sum)
  total_sum2 = np.sum(word_sum)

  context_sum = np.log2(context_sum)
  word_sum = np.log2(word_sum)

  total_sum1 = np.log2(total_sum1)
  total_sum2 = np.log2(total_sum2) 

  print(str(context_sum))
  print(str(word_sum))
  print(str(total_sum1))
  print(str(total_sum2))

  term_context_matrix = np.log2(term_context_matrix)
  term_context_matrix = term_context_matrix + total_sum1

  term_context_matrix = np.apply_along_axis(divide_array, 0, term_context_matrix, context_sum) 
  term_context_matrix = np.apply_along_axis(divide_array, 1, term_context_matrix, word_sum)
  term_context_matrix = np.clip(term_context_matrix, 0, None)
  return term_context_matrix

def create_tf_idf_matrix(term_document_matrix):
  '''Given the term document matrix, output a tf-idf weighted version.

  See section 15.2.1 in the textbook.
  
  Hint: Use numpy m atrix and vector operations to speed up implementation.

  Input:
    term_document_matrix: Numpy array where each column represents a document 
    and each row, the frequency of a word in that document.

  Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
  '''
  def count_occurence(vector, total_docs):
    doc_freq = log(np.count_nonzero(vector), 2)
    out = vector * (doc_freq - total_docs) 
    return out
  
  total_docs = term_document_matrix.shape[1]
  total_docs = log(total_docs, 2)
  term_document_matrix = np.apply_along_axis(count_occurence, 1, term_document_matrix, total_docs)
  return term_document_matrix

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  products = vector1 * vector2

  vector1 = np.square(vector1)
  vector2 = np.square(vector2)

  mag1 = sqrt(np.sum(vector1))
  mag2 = sqrt(np.sum(vector2))

  mags = mag1 * mag2

  return np.sum(products)/mags

def compute_jaccard_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  minimums = np.minimum(vector1, vector2)
  maximums = np.maximum(vector1, vector2) 
  return np.sum(minimums)/np.sum(maximums)

def compute_dice_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  minimums = np.minimum(vector1, vector2)
  denom = vector1 + vector2
  return 2 * np.sum(minimums) / np.sum(denom)

def rank_plays(target_play_index, term_document_matrix, similarity_fn):
  ''' Ranks the similarity of all of the plays to the target play.

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

  Inputs:
    target_play_index: The integer index of the play we want to compare all others against.
    term_document_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index
  '''
  
  target_vector = term_document_matrix[:, target_play_index]
  similarities = np.apply_along_axis(similarity_fn, 0, term_document_matrix, target_vector)
  #excludes the most similar vector, which should be the target vector 
  sorted_similarities = np.argsort(similarities)[::-1]
  sorted_similarities = sorted_similarities.tolist()
  return sorted_similarities

def rank_words(target_word_index, matrix, similarity_fn):
  ''' Ranks the similarity of all of the words to the target word.

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

  Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
    similarity_fn: Function that should be used to compared vectors for two word
      ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the 
    target word indexed by word_index
  '''

  target_vector = matrix[target_word_index, :]
  similarities = np.apply_along_axis(similarity_fn, 1, matrix, target_vector)
  #excludes the most similar vector, which should be the target vector 
  sorted_similarities = np.argsort(similarities)[::-1]
  sorted_similarities = sorted_similarities.tolist()
  return sorted_similarities


if __name__ == '__main__':
  tuples, document_names, vocab = read_in_shakespeare()

  print('Computing term document matrix...')
  td_matrix = create_term_document_matrix(tuples, document_names, vocab)

  print('Computing tf-idf matrix...')
  tf_idf_matrix = create_tf_idf_matrix(td_matrix)

  print('Computing term context matrix...')
  tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

  print('Computing PPMI matrix...')
  PPMI_matrix = create_PPMI_matrix(tc_matrix)

  random_idx = random.randint(0, len(document_names)-1)
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
    ranks = rank_plays(random_idx, td_matrix, sim_fn)
    for idx in range(0, 10):
      doc_id = ranks[idx]
      print('%d: %s' % (idx+1, document_names[doc_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

# used to calculate the top ten highest/lowest cosine similarity values for write up
# cos_vals = []
# for i in range(len(document_names)):
#   for j in range(i + 1, len(document_names)):
    
#     tup = (document_names[i], document_names[j], compute_cosine_similarity(td_matrix[:, i], td_matrix[:, j]))
#     cos_vals.append(tup)

# cos_vals = sorted(cos_vals, key = lambda tup: tup[2], reverse = True)
# print ("Ten Closest Plays in Vector Space")
# for i in range(0, 10):
#   print (cos_vals[i][0] + ", " + cos_vals[i][1] + ": %1.4f"   %(cos_vals[i][2]))

# print ()
# print ("Ten Most Distant Plays in Vector Space")
# for i in range(len(cos_vals)-11, len(cos_vals)-1):
#   print (cos_vals[i][0] + ", " + cos_vals[i][1] + ": %1.4f"   %(cos_vals[i][2]))
# print()

print (len(vocab))



  # word = 'juliet'
  # vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  # for sim_fn in similarity_fns:
  #   print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
  #   ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
  #   for idx in range(0, 10):
  #     word_id = ranks[idx]
  #     print('%d: %s' % (idx+1, vocab[word_id]))
