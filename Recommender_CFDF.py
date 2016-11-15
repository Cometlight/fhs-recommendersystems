# Load required modules
import csv
import numpy as np
import operator
import scipy.spatial.distance as scidist
import Simple_Recommender_CF
from geopy.distance import great_circle

# Parameters
UAM_FILE = "./data/UAM.csv"                    # user-artist-matrix (UAM)
ARTISTS_FILE = "./data/UAM_artists.csv"        # artist names for UAM
USERS_FILE = "./data/UAM_users.csv"            # user names for UAM

NEAREST_USERS = 3
MAX_ITEMS_TO_PREDICT = 10

# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:                  # open file for reading
        reader = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header
        for row in reader:
            item = row[0]
            data.append(item)
    return data





def recommender_cfdf_gender(user, UAM, max_items_to_predict, nearest_users_to_consider, users_extended):
    # user .. the user for whom we want to predict artists for
    # UAM .. user artist matrix; is modified by this function!
    # max_items_to_predict .. how many artists shall be predicted
    # nearest_users_to_consider .. how many similar users to consider
    # users_extended
    user_gender = users_extended[user][5]
    if (user_gender == 'n'):
        return Simple_Recommender_CF.simple_recommender_cf(user, UAM, max_items_to_predict, nearest_users_to_consider)

    users_different_gender_indices = np.where(users_extended[:,5] != user_gender)
    UAM[users_different_gender_indices] = 0
    return Simple_Recommender_CF.simple_recommender_cf(user, UAM, max_items_to_predict, nearest_users_to_consider)




def recommender_cfdf_age(user, UAM, max_items_to_predict, nearest_users_to_consider, users_extended):
    # user .. the user for whom we want to predict artists for
    # UAM .. user artist matrix; is modified by this function!
    # max_items_to_predict .. how many artists shall be predicted
    # nearest_users_to_consider .. how many similar users to consider
    # users_extended
    user_age = users_extended[user][1]
    if (user_age == -1):
        return Simple_Recommender_CF.simple_recommender_cf(user, UAM, max_items_to_predict, nearest_users_to_consider)

    users_extended[np.where(users_extended[:, 1] == -1)] = 999
    age_difference = np.absolute(users_extended[:, 1] - user_age)
    age_difference[user] = 999

    pc_vec = UAM[user,:]
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
       sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u,:])
    
    sort_idx = np.lexsort((sim_users * -1, age_difference.astype('int'))) # Sort by age_difference, then by sim_users*-1
    sort_idx_filtered = sort_idx[nearest_users_to_consider:]
    
    UAM[sort_idx_filtered] = 0
    return Simple_Recommender_CF.simple_recommender_cf(user, UAM, max_items_to_predict, nearest_users_to_consider)




def recommender_cfdf_country(user, UAM, max_items_to_predict, nearest_users_to_consider, users_extended):
    # user .. the user for whom we want to predict artists for
    # UAM .. user artist matrix; is modified by this function!
    # max_items_to_predict .. how many artists shall be predicted
    # nearest_users_to_consider .. how many similar users to consider
    # users_extended
    lat = users_extended[user][4]
    lon = users_extended[user][3]
    if (np.isnan(lat) or np.isnan(lon)):
        return Simple_Recommender_CF.simple_recommender_cf(user, UAM, max_items_to_predict, nearest_users_to_consider)
    
    user_coordinates = (lat, lon) # (lat, lon)
    distances_to_user = []
    for other_user_index, other_user in enumerate(users_extended):
        other_user_lat = users_extended[other_user_index][4]
        other_user_lon = users_extended[other_user_index][3]
        other_user_coordinates = (other_user_lat, other_user_lon)
        if (np.isnan(other_user_lat) or np.isnan(other_user_lon)):
            distance = 99999999999999
        else:
            distance = great_circle(user_coordinates, other_user_coordinates).miles
        distances_to_user.append(distance)
    
    distances_to_user[user] = 99999999999999

    pc_vec = UAM[user,:]
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
       sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u,:])

    sort_idx = np.lexsort((sim_users * -1, distances_to_user)) # Sort by age_difference, then by sim_users*-1
    sort_idx_filtered = np.setdiff1d(sort_idx[nearest_users_to_consider:], user)

    UAM[sort_idx_filtered] = 0
    return Simple_Recommender_CF.simple_recommender_cf(user, UAM, max_items_to_predict, nearest_users_to_consider)

