# Implementation of a simple user-based CF recommender
__author__ = 'mms'

# Load required modules
import csv
import numpy as np


# Parameters
UAM_FILE = "./data/UAM_100.txt"                    # user-artist-matrix (UAM)
ARTISTS_FILE = "./data/UAM_artists_100.txt"        # artist names for UAM
USERS_FILE = "./data/UAM_users_100.txt"            # user names for UAM

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

def simple_recommender_cf(user, UAM, max_items_to_predict, nearest_users_to_consider):
    # user .. the user for whom we want to predict artists for
    # UAM .. user artist matrix
    # max_items_to_predict .. how many artists shall be predicted
    # nearest_users_to_consider .. how many similar users to consider
    pc_vec = UAM[user,:]

    # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is already normalized)
    sim_users = np.inner(pc_vec, UAM)     # similarities between u and other users
    # print sim_users

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)        # sort in ascending order
#        print sort_idx

    # Select the closest neighbor to seed user u (which is the last but one; last one is user u herself!)
    #neighbor_idx = sort_idx[-2:-1][0] <- original TODO delete
    neighbors_idx = sort_idx[-(nearest_users_to_consider+2):-2]
    # for neighbor_idx in neighbors_idx:  # TODO beautify print output
        # print "The closest user to user " + str(user) + " are " + str(neighbor_idx) + "."
        # print "The closest user to user " + users[u] + " is user " + users[neighbor_idx] + "."

    # Get np.argsort(sim_users)l artist indices user u and her closest neighbor listened to, i.e., element with non-zero entries in UAM
    artist_idx_u = np.nonzero(UAM[user,:])                 # indices of artists user u listened to
    user_artists_idx_n = { }
    artists = []
    for neighbor_idx in neighbors_idx:
        artist_idx_n = np.nonzero(UAM[neighbor_idx,:])[0].tolist()
        user_artists_idx_n[neighbor_idx] = artist_idx_n
        artists += artist_idx_n
    # artist_idx_n = np.nonzero(UAM[neighbor_idx,:])     # indices of artists user u's neighbor listened to TODO delete
    artists = np.unique(artists)
    artists = np.setdiff1d(artists, artist_idx_u)

    artists_score = {}
    for artist in artists:
        user_artist_count = 0
        for neighbor_idx in neighbors_idx:
            playcount = UAM[neighbor_idx, artist]
            score = playcount * sim_users[neighbor_idx]
            if artist in artists_score:
                artists_score[artist] += score
            else:
                artists_score[artist] = score
            if playcount > 0:
                user_artist_count += 1

        artists_score[artist] *= float(user_artist_count) / len(neighbors_idx)

    # calculated factors of the artists' scores:
        # user similarity
        # playcount of artist
        # number of users per artist
    # could be further adjusted by adding additional weights

    sorted_recommended_artists = sorted(artists_score, key=artists_score.__getitem__, reverse=True)[:max_items_to_predict]


    # for artist, score in artists_score.iteritems():
        # print str(artist) + ": " + str(score)
    
    return sorted_recommended_artists

# Main program
if __name__ == '__main__':

    # Initialize variables
#    artists = []            # artists
#    users = []              # users
#    UAM = []                # user-artist-matrix

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
#    print users
#    print artists

    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    # For all users
    for user in range(0, UAM.shape[0]):
        # get playcount vector for current user u
        print "Next user recommendations: "
        user_most_listened_to_artists = np.argsort(UAM[user,:])[::-1]
        for i in range(0, 10):
            print str(i+1) + ". rank: " + artists[user_most_listened_to_artists[i]]
        print "- - - - - - - "
        recommended_artists_idx = simple_recommender_cv(user, UAM, MAX_ITEMS_TO_PREDICT, NEAREST_USERS)
        for i in range(0, len(recommended_artists_idx)):
            print str(i+1) + ". rank: " + artists[recommended_artists_idx[i]]
        print "\n"
