import numpy as np
import scipy as sp
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from scipy import linalg
import re
import copy
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from porter2stemmer import Porter2Stemmer
stemmer = Porter2Stemmer()
#print(stemmer.stem('conspicuous'))si
def build_matrix(docs):
    r""" Build sparse matrix from a list of documents,
    each of which is a list of word/terms in the document.
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    # Remove all ratings
    for d in docs:
        #d = d[1:]
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
    print nrows
    print ncols
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        #d = d[1:]
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()

    return mat

def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf.
    Returns scaling factors as dict. If copy is True,
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm.
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum

    if copy is True:
        return mat





k = [5]

k_values = k
train_file = "train.dat"
test_file="test.dat"
output_file = "format.dat"


print 'Working on training set'
i=0
with open(train_file, "r") as fh:
	lines = fh.readlines()
	i = i + 1

train_labels = [int(l[:2]) for l in lines]
print "train_labels"
train_text_documents = [re.sub(r'[^\w]', ' ',l[2:].lower()).split() for l in lines]
print "train_docs"

train_reviews_stem = [[t for t in d if len(t) >= 2 ] for d in train_text_documents]
train_reviews = [[stemmer.stem(t) for t in d ] for d in train_reviews_stem]

for t in train_reviews:
        new_list = get_k_mers(t)
        t.extend(new_list)

training_data = len(train_reviews)

print 'Working on  test file'
with open(test_file, "r") as fh:
    test_lines = fh.readlines()
        
text_text_documents = [re.sub(r'[^\w]', ' ',l.lower()).split() for l in test_lines]
test_reviews_stem = [[t for t in d if len(t) >= 2 ] for d in text_text_documents]
test_reviews = [[stemmer.stem(t) for t in d ] for d in test_reviews_stem]

for t in test_reviews:
        new_list = get_k_mers(t)
        t.extend(new_list)
    
testing_data = len(test_reviews)

train_reviews.extend(test_reviews)

   
print 'Building CSR matrix'
 
csr_mat = build_matrix(train_reviews)

mat_idf = csr_idf(csr_mat, copy=True)
mat_normalize = csr_l2normalize(mat_idf, copy=True)

print 'Calculate Cosine Similarity'
similarities_sparse = cosine_similarity(mat_normalize,dense_output=False)

print 'Finally, caluclating nearest neighbours'
   
all_test_labels = []

test_labels = []
for test_review_index in range(training_data, training_data +testing_data):
    similarity = similarities_sparse[test_review_index, :training_data].toarray().tolist()[0]
    similarity_with_labels = zip(similarity, train_labels, range(len(train_labels)))

    sorted_similarity_with_labels = sorted(similarity_with_labels, key=lambda (val, k, l): val, reverse=True)
           
    tmp = 0

    for j in range(k):
        if sorted_similarity_with_labels[j][0] != 0:
            tmp += int(sorted_similarity_with_labels[j][1])
        if tmp == 0:
            while tmp == 0:
                tmp = np.random.randint(-1,2)
    if tmp > 0:
        test_labels.append(1)
        tst = 1
    else:
        test_labels.append(-1)
        tst = -1


all_test_labels.append(test_labels)

   
output = open(output_file, 'w')
for t in all_test_labels[0]:
    if t == 1:
        output.write("+1")
    else:
        output.write("-1")
    output.write("\n")
output.close()



  

   

