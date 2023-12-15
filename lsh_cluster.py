import numpy as np
import pandas as pd
import re
import json
import sympy
import itertools
from random import randint
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

###U UPLOAD DATA ###
def load_data():
    json_file_path = "TVs-all-merged.json"
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

### PROCESS AND CLEAN THE DATA ###
def process_data(data):
    # Create a data frame from list of entries
    entries = [entry for model_id, entry_list in data.items() for entry in entry_list]
    old_df = pd.DataFrame(entries)

    # Check if 'featuresMap' column exists  DF, flateten the features, add to the DF
    if 'featuresMap' in old_df.columns:
        df_features = pd.json_normalize(old_df['featuresMap'])
        df = pd.concat([old_df.drop('featuresMap', axis=1), df_features], axis=1)
    else:
        df = old_df
    return df

def clean_title(title):
    if not isinstance(title, str):
        return title
    #Make the necessary data cleaning
    res = title.lower()
    hz_words = ['hertz', '-hz', '-hertz', ' hz', ' hertz']
    inch_words = ['inches', '-inch', '-inches', ' inch', ' inches', '"', ' "']
    for word in hz_words:
        res = res.replace(word, 'hz')
    for word in inch_words:
        res = res.replace(word, 'inch')
    return res

### CREATE TRAINING AND TEST DATA SETS FOR BOOTSRTRAPPING ##
def custom_train_test_split(data, train_percentage=0.63, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    num_samples = len(data)
    num_train_samples = int(num_samples * train_percentage)
    
    # Randomly shuffle the indices
    shuffled_indices = np.random.permutation(num_samples)
    # Split the indices into training and test sets
    train_indices = shuffled_indices[:num_train_samples]
    test_indices = shuffled_indices[num_train_samples:]
    
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    
    return train_data, test_data

### MODEL WORDS AND BINARY MATRIX ###
def get_model_words(model_name):
    #regex function
    pattern = r'\b(?=\w*[a-zA-Z])(?=\w*[0-9])(?=\w*[^\s])\w+\b'
    return re.findall(pattern, model_name)

def  create_binary_matrix(df, column_name, get_model_words):
    # Collecting model words from titles
    model_words = set()
    for model_name in df[column_name]:
        model_words.update(get_model_words(model_name))
    model_words = list(model_words)

    # Creating the binary matrix
    binary_matrix = np.zeros((len(model_words), len(df)))
    for i, word in enumerate(model_words):
        for j, title in enumerate(df[column_name]):
            binary_matrix[i, j] = int(" " + word in title)
    
    return binary_matrix, model_words

### MINHASHING ###
#normal minhash algorithm which I did not end up using
def minhash(input_matrix, num_permutations, seeds = None):
    num_rows = len(input_matrix)
    num_columns = len(input_matrix[0])
    signature_matrix = np.zeros((num_permutations, num_columns))
    for i in range(num_permutations):
        if seeds is not None:
            rng = np.random.default_rng(seeds[i])
        else:
            rng = np.random.default_rng()
        perm = rng.permutation(num_rows)
        for k in range(num_columns):
            j = 0
            while not input_matrix[perm[j], k]:
                j += 1
                if j == len(input_matrix):
                    break
            signature_matrix[i, k] = j
    return signature_matrix

#fast minhash algorithm
def fast_minhash(input_matrix, num_hash_functions):
    N = num_hash_functions
    cols = input_matrix.shape[1]
    max_val = (2**32)-1
    perms = [ (randint(0,max_val), randint(0,max_val)) for _ in range(N)]
    prime = sympy.nextprime(input_matrix.shape[0] + 1)
    #create hash function
    hash_functions = [lambda x: (x * perms[i][0] + perms[i][1]) % prime for i in range(N)]
    
    # Initializing signature matrix
    M = np.full((N, cols), prime + 1)
    
    for r in range(input_matrix.shape[0]):
        hashes = []
        for perm in perms:
            h = (perm[0] * r + perm[1]) % prime
            hashes.append(h)
        for c in range(cols):
            if input_matrix[r, c] == 1:
                for i in range(N):
                    if hashes[i] < M[i, c]:
                        M[i, c] = hashes[i]    
    return M


### DEFINE THE POSSIBLE RANGE OF BANDS AND ROWS ###
def choose_b_r(matrix):
    numHashes, numProducts=matrix.shape
    rows=[]
    bands=[]
    for r in range(numHashes+1):
        for b in range (numHashes+1):
            if b*r==numHashes:
                rows.append(r)
                bands.append(b)    
    return rows, bands

### LSH ###
def LSH(signature_matrix, b, r):
    # Initialize a dictionary for buckets
    buckets = dict()
    for k in range(b):
        # For each column in the signature matrix
        for j in range(signature_matrix.shape[1]):
            # Extract the current band for this column
            band = signature_matrix[int(k * r) : int((k + 1) * r), j].astype(int)
            # Create a hash value for the band and add index to buckets
            h = hash(tuple(band))
            if h in buckets:
                buckets[h].add(j)
            else:
                buckets[h] = set([j])

    return buckets

#Get the combinations of 2 from the same bucket if ther are more elements than one
def get_candidate_pairs(hashedbuckets):
    LSH_pairs = set()
    # Iterate buckets
    for bucket_pairs in hashedbuckets.values():
        if len(bucket_pairs) > 1:
            # Create combinations
            for pair in itertools.combinations(bucket_pairs, 2):  
                LSH_pairs.add(pair)
                
    # Convert to a list
    lsh_list = list(LSH_pairs)
    # Remove the pair if the same pair is added twice
    lsh_list_new = [el for el in lsh_list if el[0] < el[1]]
    LSH_pairs_new = set(lsh_list_new)
    
    return LSH_pairs_new


### COUNT THE REAL NUMBER OF DUCPLICATES IN THE WHOLE DATA SET ###
def count_real_duplicates(df):
    grouped = df.groupby('modelID').size()
    duplicates = grouped[grouped > 1]
    num_real_duplicates = sum(d * (d - 1) // 2 for d in duplicates)
    return num_real_duplicates

### CALCULATE JACCARD SIMILARITY ###
def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0

### REDEFINE TEHE LIST OF CANDIDATE PAIRS ###
def redefine_candidate_pairs(candidate_pairs, binary_matrix, model_words, threshold=0.5):
    refined_pairs = []
    for i, j in candidate_pairs:
        set_i = set(model_words[k] for k, present in enumerate(binary_matrix[:, i]) if present)
        set_j = set(model_words[k] for k, present in enumerate(binary_matrix[:, j]) if present)
        similarity = calculate_jaccard_similarity(set_i, set_j)
        if similarity >= threshold:
            refined_pairs.append((i, j))
    return refined_pairs

### CALCULATE THE DISTANCE MATRIX FROM THE DISSIMILARITY (1-JACCARD) ###
def calculate_distance_matrix(candidate_pairs, binary_matrix, model_words, distance_matrix):
    for i, j in candidate_pairs:
        if i != j:
            set_i = set(model_words[k] for k, present in enumerate(binary_matrix[:, i]) if present)
            set_j = set(model_words[k] for k, present in enumerate(binary_matrix[:, j]) if present)
            similarity = calculate_jaccard_similarity(set_i, set_j)
            # Jaccard distance for dissimilarity matrix
            distance = 1 - similarity 
            distance_matrix[i, j] = distance
            # Ensure the matrix is symmetric
            distance_matrix[j, i] = distance  
    #Fill the diagonals as zeros as the distance to yourself is zero
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


#Check if the distance matrix differs
def sum_distance_rows(distance_matrix, num_rows=25):
    row_sums = np.sum(distance_matrix, axis=1)
    for i, row_sum in enumerate(row_sums[:num_rows]):
        print(f"Sum of row {i}: {row_sum}")

### FINDING THE BEST THRESHOLD ###
def find_optimal_threshold(data, distance_matrix, LSH_pairs, start=0.3, end=1.0, step=0.05):
    optimal_threshold = start
    best_f1_score = 0

    # Iterate over potential threshold values
    for threshold in np.arange(start, end + step, step):
        # Perform hierarchical clustering at the current threshold
        cluster_pairs = hierarchical_clustering(distance_matrix, threshold)
        
        # Evaluate  using F1 score
        _, _, _, _, _, f1_score, _, _, _ = evaluate_cluster_LSH(LSH_pairs, cluster_pairs, data)
        
        # Update the optimal threshold if the current F1 score is better
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            optimal_threshold = threshold

    return optimal_threshold

### PERFORM CLUSTERING ###        
def hierarchical_clustering(distance_matrix, cluster_threshold):
    condensed_distance_matrix = squareform(distance_matrix)
    Z = linkage(condensed_distance_matrix, method='complete')
    clusters = fcluster(Z, cluster_threshold, criterion='distance')

    cluster_pairs = []
    clustered_items = {i: [] for i in np.unique(clusters)}
    for i, cluster_label in enumerate(clusters):
        clustered_items[cluster_label].append(i)
    #get combinations of the cluster pairs
    for cluster in clustered_items.values():
        cluster_pairs.extend(combinations(cluster, 2))

    return cluster_pairs

### EVALUATION METRICS ##
def evaluate_cluster_LSH(LSH_pairs, cluster_pairs, data):
    actualPairs_LSH = 0
    actualPairs_cluster = 0
   
    # Count for LSH pairs
    for pair in LSH_pairs:
        if pair[0] < len(data) and pair[1] < len(data) and data.iloc[pair[0]]['modelID'] == data.iloc[pair[1]]['modelID']:
            actualPairs_LSH += 1

    # Count for Jaccard cluster pairs
    for pair in cluster_pairs:
        if pair[0] < len(data) and pair[1] < len(data) and data.iloc[pair[0]]['modelID'] == data.iloc[pair[1]]['modelID']:
            actualPairs_cluster += 1

    numberDuplicates = count_real_duplicates(data)
    N = len(data)
    totalComparisons = (N * (N - 1)) / 2
    
    # For LSH
    numberCandidatesLSH = len(LSH_pairs) if len(LSH_pairs) > 0 else 0
    PQ = actualPairs_LSH / numberCandidatesLSH
    PC = actualPairs_LSH / numberDuplicates
    F1StarLSH = 2 * (PQ * PC) / (PQ + PC) if PQ + PC > 0 else 0
    fraction_comparisonLSH = numberCandidatesLSH / totalComparisons
    
    # For Jaccard based  clustering
    PQ_cluster = actualPairs_cluster / (len(LSH_pairs)) if len(LSH_pairs) > 0 else 0
    PC_cluster = actualPairs_cluster / numberDuplicates
    precision= actualPairs_cluster/(len(cluster_pairs)) if len(cluster_pairs) > 0 else 0
    recall=actualPairs_cluster/numberDuplicates
    F1 = (2*(precision*recall))/(precision+recall) if precision+recall > 0 else 0
    F1_cluster = 2 * (PQ_cluster * PC_cluster) / (PQ_cluster + PC_cluster) if PQ_cluster + PC_cluster > 0 else 0
    fraction_cluster = len(cluster_pairs) / totalComparisons
   
    return [PQ, PC, F1StarLSH, PQ_cluster, PC_cluster, F1, F1_cluster, fraction_comparisonLSH, fraction_cluster]



