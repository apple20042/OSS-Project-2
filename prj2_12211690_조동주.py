import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

file = 'ratings.dat'
dataframe = pd.read_csv(file, sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
user = dataframe['UserID'].nunique()
movie = dataframe['MovieID'].max() 
user_item_matrix = np.zeros((user, movie)) 
for row in dataframe.itertuples():
    user_item_matrix[row[1]-1, row[2]-1] = row[3]
kmeans = KMeans(n_clusters=3, random_state=42)
user_clusters = kmeans.fit_predict(user_item_matrix)  

#AU
def AU(user_item_matrix, user_clusters, cluster_id):
    cluster_users_indices = np.where(user_clusters == cluster_id)[0]
    cluster_user_item_matrix = user_item_matrix[cluster_users_indices]
    item_sum = np.sum(cluster_user_item_matrix, axis=0)
    au_recommend = np.argmax(item_sum) + 1
    return au_recommend
print("AU")
for cluster_id in range(3):
    au_recommend = AU(user_item_matrix, user_clusters, cluster_id)
    print(au_recommend)    
print("\n")

#Avg
def Avg(user_item_matrix, user_clusters, cluster_id):
    cluster_users_indices = np.where(user_clusters == cluster_id)[0]
    cluster_user_item_matrix = user_item_matrix[cluster_users_indices]
    item_means = np.mean(cluster_user_item_matrix, axis = 0)
    avg_recommend = np.argmax(item_means) + 1
    return avg_recommend
print("Avg")
for cluster_id in range(3):
    avg_recommend = Avg(user_item_matrix, user_clusters, cluster_id)
    print(avg_recommend)
print("\n")

#SC
def SC(user_item_matrix, user_clusters, cluster_id):
    cluster_users_indices = np.where(user_clusters == cluster_id)[0]
    cluster_user_item_matrix = user_item_matrix[cluster_users_indices]
    item_counts = np.sum(cluster_user_item_matrix != 0, axis = 0)
    SC_recommend = np.argmax(item_counts) + 1
    return SC_recommend
print("SC")
for cluster_id in range(3):
    SC_recommend = SC(user_item_matrix, user_clusters, cluster_id)
    print(SC_recommend)
print("\n")

#AV
def AV(user_item_matrix, user_clusters, cluster_id, threshold=4):
    cluster_users_indices = np.where(user_clusters == cluster_id)[0]
    cluster_user_item_matrix = user_item_matrix[cluster_users_indices]
    positive_ratings_indices = np.where(cluster_user_item_matrix >= threshold)
    positive_item_scores = np.mean(cluster_user_item_matrix[positive_ratings_indices], axis = 0)
    AV_recommend = np.argmax(positive_item_scores) + 1
    return AV_recommend
print("AV")
for cluster_id in range(3):
    AV_recommend = AV(user_item_matrix, user_clusters, cluster_id)
    print(AV_recommend)
print("\n")

#BC
def BC(user_item_matrix, user_clusters, cluster_id):
    cluster_users_indices = np.where(user_clusters == cluster_id)[0]
    cluster_user_item_matrix = user_item_matrix[cluster_users_indices]
    item_ranks = np.argsort(cluster_user_item_matrix, axis = 1)
    n_items = cluster_user_item_matrix.shape[1]
    scores = np.zeros(n_items)
    for user_ratings in item_ranks:
        for rank, item_index in enumerate(user_ratings):
            scores[item_index] += (n_items - 1) - rank
    max_score = np.max(scores)
    min_score = np.min(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)
    BC_recommend = np.argsort(-normalized_scores) + 1
    return BC_recommend
print("BC")
for cluster_id in range(3):
    BC_recommend = BC(user_item_matrix, user_clusters, cluster_id)
    print(BC_recommend)
print("\n")

#CR
def CR(user_item_matrix, user_clusters, cluster_id):
    cluster_users_indices = np.where(user_clusters == cluster_id)[0]
    cluster_user_item_matrix = user_item_matrix[cluster_users_indices]
    n_items = cluster_user_item_matrix.shape[1]
    scores = np.zeros(n_items)
    for i in range(n_items):
        for j in range(i+1, n_items):
            compare_results = np.sum(cluster_user_item_matrix[:, i] > cluster_user_item_matrix[:, j])
            if compare_results > num_users / 2:
                scores[i] += 1
            elif compare_results < num_users / 2:
                scores[j] += 1
            else:
                scores[i] += 0.5
                scores[j] += 0.5

    max_score = np.max(scores)
    min_score = np.min(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)
    CR_recommend = np.argsort(-normalized_scores) + 1
    return CR_recommend
print("CR")
for cluster_id in range(3):
    CR_recommend = CR(user_item_matrix, user_clusters, cluster_id)
    print(CR_recommend)