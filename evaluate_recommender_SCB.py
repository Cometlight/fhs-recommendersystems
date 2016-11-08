# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different baseline recommenders: collaborative filtering,
# content-based recommender, random recommendation, and simple hybrid methods.
# It also implements a score-based fusion technique for hybrid recommendation.
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
from time import sleep
import pandas as pd
import math
import Simple_Recommender_CF
import Evaluate_Recommender
import operator

# Parameters
UAM_FILE = "./data/C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = "UAM_artists.txt"    # artist names for UAM
USERS_FILE = "UAM_users.txt"        # user names for UAM
AAM_FILE = "./data/C1ku_AAM.txt"                # artist-artist similarity matrix (AAM)
METHOD = "CB"                       # recommendation method
                                    # ["RB", "CF", "CB", "HR_SEB", "HR_SCB"]

NF = 10              # number of folds to perform in cross-validation
NO_RECOMMENDED_ARTISTS = 100
VERBOSE = True     # verbose output?

# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter='\t')  # create reader
        headers = reader.next()  # skip header
        for row in reader:
            item = row[0]
            data.append(item)
    f.close()
    return data


# Function that implements a CF recommender. It takes as input the UAM,
# the index of the seed user (to make predictions for) and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CF(UAM, seed_uidx, seed_aidx_train, K):
    # UAM               user-artist-matrix
    # seed_uidx         user index of seed user
    # seed_aidx_train   indices of training artists for seed user
    # K                 number of nearest neighbors (users) to consider for each seed users

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Remove information on test artists from seed's listening vector
    aidx_nz = np.nonzero(pc_vec)[0]                             # artists with non-zero listening events
    aidx_test = np.intersect1d(aidx_nz, seed_aidx_train)        # intersection between all artist indices of user and train indices gives test artist indices
    # print aidx_test

    # Set to 0 the listening events of seed user user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx, :] = UAM[seed_uidx, :] / np.sum(UAM[seed_uidx, :])

    # Compute similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
        sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u,:])

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)  # sort in ascending order

    # Select the closest neighbor to seed user (which is the last but one; last one is user u herself!)
    neighbor_idx = sort_idx[-1-K:-1]

    # Get all artist indices the seed user and her closest neighbor listened to, i.e., element with non-zero entries in UAM
    artist_idx_u = seed_aidx_train                      # indices of artists in training set user
    # for k=1:
    # artist_idx_n = np.nonzero(UAM[neighbor_idx, :])     # indices of artists user u's neighbor listened to
    # for k>1:
    artist_idx_n = np.nonzero(UAM[neighbor_idx, :])[1]    # [1] because we are only interested in non-zero elements among the artist axis

    # Compute the set difference between seed user's neighbor and seed user,
    # i.e., artists listened to by the neighbor, but not by seed user.
    # These artists are recommended to seed user.
    recommended_artists_idx = np.setdiff1d(artist_idx_n, artist_idx_u)


    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}           # dictionary to hold recommended artists and corresponding scores
    # Compute artist scores. Here, just derived from max-to-1-normalized play count vector of nearest neighbor (neighbor_idx)
    # for k=1:
    # scores = UAM[neighbor_idx, recommended_artists_idx] / np.max(UAM[neighbor_idx, recommended_artists_idx])
    # for k>1:
    scores = np.mean(UAM[neighbor_idx][:, recommended_artists_idx], axis=0)

    # Write (artist index, score) pairs to dictionary of recommended artists
    for i in range(0, len(recommended_artists_idx)):
        dict_recommended_artists_idx[recommended_artists_idx[i]] = scores[i]
    #########################################


    # Return dictionary of recommended artist indices (and scores)
    return dict_recommended_artists_idx


# Function that implements a content-based recommender. It takes as input an artist-artist-matrix (AAM) containing pair-wise similarities
# and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CB(AAM, seed_aidx_train, K, no_recommendations):
    # AAM                 artist-artist-matrix of pairwise similarities
    # seed_aidx_train     indices of training artists for seed user
    # K                   number of nearest neighbors (artists) to consider for each seed artist
    # no_recommendations  max number of recommended artists; no_recommendations <= K

    if no_recommendations > K:
        print "K must be greater than (or equal to) no_recommendations"
        return

    print "K: " + str(K)
    print "no_recommendations: " + str(no_recommendations)

    # Get nearest neighbors of train set artist of seed user
    # Sort AAM column-wise for each row
    sort_idx = np.argsort(AAM[seed_aidx_train, :], axis=1)

    # Select the K closest artists to all artists the seed user listened to
    neighbor_idx = sort_idx[:,-1-K:-1]


    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}           # dictionary to hold recommended artists and corresponding scores

    # Distill corresponding similarity scores and store in sims_neighbors_idx
    sims_neighbors_idx = np.zeros(shape=(len(seed_aidx_train), K), dtype=np.float32)
    for i in range(0, neighbor_idx.shape[0]):
        sims_neighbors_idx[i] = AAM[seed_aidx_train[i], neighbor_idx[i]]

    # Aggregate the artists in neighbor_idx.
    # To this end, we compute their average similarity to the seed artists
    uniq_neighbor_idx = set(neighbor_idx.flatten())     # First, we obtain a unique set of artists neighboring the seed user's artists.
    # Now, we find the positions of each unique neighbor in neighbor_idx.

    for nidx in uniq_neighbor_idx:

        mask = np.where(neighbor_idx == nidx)
        #print mask
        # Apply this mask to corresponding similarities and compute average similarity
        avg_sim = np.mean(sims_neighbors_idx[mask])
        # Store artist index and corresponding aggregated similarity in dictionary of artists to recommend
        # Length of mask[0] = count of recommendations for this artist

        # Normalization
        # scores can be normalized like this:  score_normalized = (score - min) / (max - min)
        # score calculation for our CB is:      dict_recommended_artists_idx[nidx] = avg_sim * len(mask[0])
        # so the max possible score should be:                                 max =   1     * NO_RECOMMENDED_ARTISTS
        # the min score should be                                              min =   0     * 0

        score = avg_sim * len(mask[0])
        max_score = 1 * no_recommendations
        min_score = 0 
        score_normalized = (score - min_score) / (max_score - min_score) 

        dict_recommended_artists_idx[nidx] = score_normalized

    #########################################

    # Remove all artists that are in the training set of seed user
    for aidx in seed_aidx_train:
        dict_recommended_artists_idx.pop(aidx, None)            # drop (key, value) from dictionary if key (i.e., aidx) exists; otherwise return None

    # Sort dictionary by similarity; returns list of tuples(artist_idx, sim)
    recommendations = sorted(dict_recommended_artists_idx.items(), key=operator.itemgetter(1), reverse=True)[:no_recommendations]
    # recommendations = sorted([(key,value) for (key,value) in dict_recommended_artists_idx.items()], reverse=False)[:no_recommendations]

    # Return sorted list of recommended artist indices (and scores)
    return dict(recommendations)


# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx


# Function to run an evaluation experiment.
def run():
    # Initialize variables to hold performance measures
    avg_prec = 0;       # mean precision
    avg_rec = 0;        # mean recall

    # For all users in our data (UAM)
    no_users = 10#UAM.shape[0]
    no_artists = UAM.shape[1]

    # Init sparse user count
    no_sparse_users = 0
    no_sparse_folds = 0

    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0] 

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0
        
        # ignore sparse users
        if len(u_aidx) < NF:
            no_sparse_users += 1
            continue

        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV
        for train_aidx, test_aidx in kf:  # for all folds

            # Show progress
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(len(train_aidx)) + ", Test items: " + str(len(test_aidx)),      # the comma at the end avoids line break

            # Call recommend function
            copy_UAM = UAM.copy()       # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable
            if not Evaluate_Recommender.create_training_UAM(copy_UAM, u, train_aidx):
                no_sparse_folds = no_sparse_folds + 1
                continue

            # Run recommendation method specified in METHOD
            # NB: u_aidx[train_aidx] gives the indices of training artists
            #K_RB = 10          # for RB: number of randomly selected artists to recommend
            #K_CB = 3           # for CB: number of nearest neighbors to consider for each artist in seed user's training set
            #K_CF = 3           # for CF: number of nearest neighbors to consider for each user
            #K_HR = 10          # for hybrid: number of artists to recommend at most

            if METHOD == "RB":          # random baseline
                dict_rec_aidx = recommend_RB(np.setdiff1d(range(0, no_artists), u_aidx[train_aidx]), NO_RECOMMENDED_ARTISTS) # len(test_aidx))

            elif METHOD == "CF":        # collaborative filtering
                dict_rec_aidx = Simple_Recommender_CF.simple_recommender_cf(u, copy_UAM, NO_RECOMMENDED_ARTISTS, K_CF)

            elif METHOD == "CB":        # content-based recommender
                dict_rec_aidx = recommend_CB(AAM, u_aidx[train_aidx], K_CB, NO_RECOMMENDED_ARTISTS)

            elif METHOD == "HR_SCB":     # hybrid of CF and CB, using score-based fusion (SCB)
                dict_rec_aidx_CB = recommend_CB(AAM, u_aidx[train_aidx], K_CB, NO_RECOMMENDED_ARTISTS)
                dict_rec_aidx_CF = Simple_Recommender_CF.simple_recommender_cf(u, copy_UAM, NO_RECOMMENDED_ARTISTS, K_CF)

                # Original way of aggregating, before rank aggregation was introduced:
                # weight_CB = 1
                # weight_CF = 1
                #
                # scores_fused = {}
                # no_recommendations = int(math.floor((K_CB + K_CF) / 2))
                #
                # print no_recommendations
                #
                # dict_rec_aidx = {}
                #
                # for aidx in dict_rec_aidx_CB.keys():
                #     scores_fused[aidx] = weight_CB * dict_rec_aidx_CB[aidx]**2
                #
                # for aidx in dict_rec_aidx_CF.keys():
                #     if aidx in scores_fused:
                #         scores_fused[aidx] += weight_CF * dict_rec_aidx_CF[aidx]**2
                #     else:
                #         scores_fused[aidx] = weight_CF * dict_rec_aidx_CF[aidx]**2
                #
                # sorted_rec_aidx = sorted([(key,value) for (key,value) in scores_fused.items()], reverse=False)[:no_recommendations]
                # dict_rec_aidx = dict(sorted_rec_aidx)

                # Rank aggregation (Borda rank count):
                

                # sorted by ascending artist score ... highest index, will be the one with the highest score, so this one will get the most votes
                cb_recommended_artists_sorted_by_value = sorted(dict_rec_aidx_CB.keys(), key=dict_rec_aidx_CB.get)
                cf_recommended_artists_sorted_by_value = sorted(dict_rec_aidx_CF.keys(), key=dict_rec_aidx_CF.get)

                votes_final = {} # Key: Artist, Value: Number of votes

                # add votes from CB
                # enumerate returns the index within the for loop (which in our case == votes) and also the value (which in our case == artist id)
                for votes, artist in enumerate(cb_recommended_artists_sorted_by_value): 
                    votes_final[artist] = votes + 1

                # add votes from CF
                for votes, artist in enumerate(cf_recommended_artists_sorted_by_value): 
                    if artist not in votes_final:
                        votes_final[artist] = votes + 1
                    else:
                        votes_final[artist] += (votes + 1)


                no_recommendations = min(len(votes_final), NO_RECOMMENDED_ARTISTS)
                sorted_rec_aidx = sorted(votes_final.items(), key=operator.itemgetter(1), reverse=True)[:no_recommendations]
                # sorted_rec_aidx = sorted([(key,value) for (key,value) in votes_final.items()], reverse=False)[:no_recommendations]
                dict_rec_aidx = dict(sorted_rec_aidx)


            rec_aidx = dict_rec_aidx.keys()

            if VERBOSE:
                print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)          # correctly predicted artists
            # True Positives is amount of overlap in recommended artists and test artists
            TP = len(correct_aidx)
            # False Positives is recommended artists minus correctly predicted ones
            FP = len(np.setdiff1d(rec_aidx, correct_aidx))

            # Precision is percentage of correctly predicted among predicted
            # Handle special case that not a single artist could be recommended -> by definition, precision = 100%
            if len(rec_aidx) == 0:
                prec = 100.0
            else:
                prec = 100.0 * TP / len(rec_aidx)

            # Recall is percentage of correctly predicted among all listened to
            # Handle special case that there is no single artist in the test set -> by definition, recall = 100%
            if len(test_aidx) == 0:
                rec = 100.0
            else:
                rec = 100.0 * TP / len(test_aidx)


            # add precision and recall for current user and fold to aggregate variables
            avg_prec += prec
            avg_rec += rec

            # Output precision and recall of current fold
            if VERBOSE:
                print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

            # Increase fold counter
            fold += 1

    avg_prec /= (NF * (no_users - no_sparse_users - no_sparse_folds/NF))
    avg_rec /= (NF * (no_users - no_sparse_users - no_sparse_folds/NF))
    # Output mean average precision and recall
    f1_score = 2 * ( (avg_prec * avg_rec) / (avg_prec + avg_rec))

    # Output mean average precision and recall
    if VERBOSE:
        print ("\nMAP: %.2f, MAR: %.2f, F1 Score: %.2f" % (avg_prec, avg_rec, f1_score))
        print str(no_sparse_folds)


# Main program, for experimentation.
if __name__ == '__main__':

    # Load metadata from provided files into lists
    #artists = read_from_file(ARTISTS_FILE)
    #users = read_from_file(USERS_FILE)
    # Load UAM
    print "Loading UAM... ",
    # UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
    UAM = pd.read_csv(UAM_FILE, delimiter='\t', dtype=np.float32).values # greatly increase reading speed via pandas
    print "Done."
    # Load AAM
    print "Loading AAM... ",
    # AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)
    AAM = pd.read_csv(AAM_FILE, delimiter='\t', dtype=np.float32).values # greatly increase reading speed via pandas
    print "Done."
    
    if False:
        METHOD = "HR_SCB"
        print METHOD
        K_CB = NO_RECOMMENDED_ARTISTS     # number of nearest neighbors to consider in CB (= artists)
        K_CF = 25                         # number of nearest neighbors to consider in CF (= users)
        for K_HR in range(10, 11):
            print (str(K_HR) + ","),
            run()
        # NO_RECOMMENDED_ARTISTS = 1: 
        # NO_RECOMMENDED_ARTISTS = 5: 
        # NO_RECOMMENDED_ARTISTS = 10: Stephan
        # NO_RECOMMENDED_ARTISTS = 20:  
        # NO_RECOMMENDED_ARTISTS = 50: Dani
        # NO_RECOMMENDED_ARTISTS = 75:  
        # NO_RECOMMENDED_ARTISTS = 100: Lukas

    if True:
        METHOD = "CB"
        print METHOD
        K_CB = NO_RECOMMENDED_ARTISTS
        print (str(K_CB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1: 
        # NO_RECOMMENDED_ARTISTS = 5: 
        # NO_RECOMMENDED_ARTISTS = 10: 
        # NO_RECOMMENDED_ARTISTS = 20: LUKAS
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 1.46, MAR: 3.22, F1 Score: 2.01
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 1.40, MAR: 4.55, F1 Score: 2.15
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 1.36, MAR: 5.66, F1 Score: 2.20

    if False:
        METHOD = "CF"
        print METHOD
        K_CF = 25
        print (str(K_CF) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1: 
        # NO_RECOMMENDED_ARTISTS = 5: 
        # NO_RECOMMENDED_ARTISTS = 10: Lukas
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 3.67, MAR: 2.80, F1 Score: 3.18 (1349 sparse_folds)
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 3.10, MAR: 5.93, F1 Score: 4.07 (1349 sparse_folds)
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 2.84, MAR: 8.00, F1 Score: 4.19   (1349 sparse_folds)
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 2.65, MAR: 9.88, F1 Score: 4.18  (1349 sparse_folds)

    if False:
        METHOD = "RB"
        print METHOD
        K_RB = NO_RECOMMENDED_ARTISTS
        print (str(K_RB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1: 
        # NO_RECOMMENDED_ARTISTS = 5: Lukas
        # NO_RECOMMENDED_ARTISTS = 10: MAP: 0.35, MAR: 0.50, F1 Score: 0.41 (1349 sparse_folds)
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 0.35, MAR: 0.20, F1 Score: 0.26 (1349 sparse_folds)
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 0.37, MAR: 0.11, F1 Score: 0.16 (1349 sparse_folds)
        # NO_RECOMMENDED_ARTISTS = 75: 
        # NO_RECOMMENDED_ARTISTS = 100: Stephan
