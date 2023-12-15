from lsh_cluster import *

data = load_data()
df = process_data(data)
df['title'] = df['title'].apply(clean_title)
df.iloc[:, 4:] = df.iloc[:, 4:].applymap(clean_title)
numBS=5

for boot in range(numBS): 
    train_data, test_data = train_test_split(df, train_percentage=0.63, random_seed=42)

    binary_matrix, model_words = create_binary_matrix(test_data, 'title', get_model_words)
    
    num_hash=500
    signature_matrix=fast_minhash(binary_matrix, num_hash)
    range_b, range_r = choose_b_r(signature_matrix)
    
    result = pd.DataFrame(columns=['PQ', 'PC', 'F1StarLSH', 'PQ_cluster', 'PC_cluster', 'F1', 'F1_cluster', 'fraction_comparisonLSH', 'fraction_cluster', 'precision', 'recall','b' ,'r', 'thres'])
    for i in range(len(range_b)):
        b=range_b[i]
        r=range_r[i]
        
        bucket_list=LSH(signature_matrix, b, r)
        LSH_pairs=get_candidate_pairs(bucket_list)
        
        num_items = len(test_data)
        distance_matrix = np.ones((num_items, num_items))
        distance_matrix=calculate_distance_matrix(LSH_pairs, binary_matrix, model_words, distance_matrix)
    
        threshold= find_optimal_threshold(test_data, distance_matrix, LSH_pairs, start=0.3, end=1.0, step=0.05)

        cluster_pairs=hierarchical_clustering(distance_matrix, threshold)
        
        metrics=evaluate_cluster_LSH(LSH_pairs, cluster_pairs, test_data)
        metrics.append(b)
        metrics.append(r)
        metrics.append(threshold)
        
        result.loc[len(result.index)]=evals
        
    print(f"bootstrap done: {boot}")
    result.to_excel("result"+ str(boot) + ".xlsx")

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

file_paths = [
    'C:/Users/dogaa/OneDrive/Desktop/result0.xlsx',
    'C:/Users/dogaa/OneDrive/Desktop/result1.xlsx',
    'C:/Users/dogaa/OneDrive/Desktop/result2.xlsx',
    'C:/Users/dogaa/OneDrive/Desktop/result3.xlsx',
    'C:/Users/dogaa/OneDrive/Desktop/result4.xlsx'
]

result = [pd.read_excel(path) for path in file_paths]
combine = pd.concat(result)
average = combine.groupby(combine['fraction_comparisonLSH']).mean().reset_index()

color_cycle = cycler(color=['#add8e6', '#1f77b4'])

x_min = -0.05  
x_max = 1

# Update default rc settings
plt.rcParams.update({
    'axes.prop_cycle': color_cycle,
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'grid.color': 'gray',
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.labelsize': 'large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'legend.fontsize': 'medium',
    'figure.dpi': 300
})


# Plot Pair Completeness (PC) for clustering and LSH
plt.figure(figsize=(20, 8))
plt.plot(average['fraction_comparisonLSH'], average['PC_cluster'], label='PC Clustering')
plt.plot(average['fraction_comparisonLSH'], average['PC'], label='PC LSH', linestyle='--')
plt.xlabel("Fraction of comparisons")
plt.ylabel("Pair Completeness")
plt.legend()
plt.title("Pair Completeness")
plt.xlim(x_min, x_max)
plt.show()

# Plot Pair Quality (PQ) for clustering and LSH
plt.figure(figsize=(20, 8))
plt.plot(average['fraction_comparisonLSH'], average['PQ_cluster'], label='PQ Clustering')
plt.plot(average['fraction_comparisonLSH'], average['PQ'], label='PQ LSH', linestyle='--')
plt.xlabel("Fraction of comparisons")
plt.ylabel("Pair Quality")
plt.legend()
plt.title("Pair Quality")
plt.xlim(x_min, x_max)
plt.show()

# Plot F1 Score for clustering and LSH
plt.figure(figsize=(20, 8))
plt.plot(average['fraction_comparisonLSH'], average['F1'], label='F1 Clustering')
plt.plot(average['fraction_comparisonLSH'], average['F1StarLSH'], label='F1 LSH', linestyle='--')
plt.xlabel("Fraction of comparisons")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1 Score")
plt.xlim(x_min, x_max)
plt.show()
        