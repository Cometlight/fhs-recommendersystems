# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different baseline recommenders: collaborative filtering,
# content-based recommender, random recommendation, and simple hybrid methods.
# It also implements a score-based fusion technique for hybrid recommendation.__author__ = 'mms'

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
import Evaluate_Recommender_Extended
import Recommender_CFDF
import random

# Parameters
UAM_FILE = "./data/C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = "UAM_artists.txt"    # artist names for UAM
USERS_FILE = "UAM_users.txt"        # user names for UAM
AAM_FILE = "./data/C1ku_AAM.txt"                # artist-artist similarity matrix (AAM)
METHOD = "CB"                       # recommendation method
                                    # ["RB", "CF", "CB", "HR_SEB", "HR_SCB"]
USERS_EXTENDED_FILE = "./data/C1ku_users_extended.csv"

NF = 10              # number of folds to perform in cross-validation
NO_RECOMMENDED_ARTISTS = 300
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
    aidx_test = np.setdiff1d(aidx_nz, seed_aidx_train)        # intersection between all artist indices of user and train indices gives test artist indices
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
    no_users = UAM.shape[0]
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
            if not Evaluate_Recommender.create_training_UAM(copy_UAM, u, u_aidx[train_aidx]):
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

            if METHOD == "RB_U":          # random user baseline
                dict_rec_aidx = Evaluate_Recommender_Extended.recommend_RB_user(copy_UAM, u_aidx[train_aidx], NO_RECOMMENDED_ARTISTS)

            elif METHOD == "CF":        # collaborative filtering
                dict_rec_aidx = Simple_Recommender_CF.simple_recommender_cf(u, copy_UAM, NO_RECOMMENDED_ARTISTS, K_CF)

            elif METHOD == "CB":        # content-based recommender
                dict_rec_aidx = recommend_CB(AAM, u_aidx[train_aidx], K_CB, NO_RECOMMENDED_ARTISTS)

            elif METHOD == "HR_BRB":     # hybrid of CF and CB, using score-based fusion (SCB)
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


            elif METHOD == "PB":  # content-based recommender
                dict_rec_aidx = Evaluate_Recommender_Extended.recommend_PB(copy_UAM, u_aidx[train_aidx], NO_RECOMMENDED_ARTISTS)

            elif METHOD == "HR_SEB": # hybrid, set-based
                rec_aidx_CF = Simple_Recommender_CF.simple_recommender_cf(u, copy_UAM, NO_RECOMMENDED_ARTISTS, K_CF)
                rec_aidx_CB = recommend_CB(AAM, u_aidx[train_aidx], K_CB, NO_RECOMMENDED_ARTISTS)
                rec_aidx_seb = np.intersect1d(rec_aidx_CB, rec_aidx_CF)[:NO_RECOMMENDED_ARTISTS]  # Perform "set-based fusion". It's as simple as this.

                i = 0

                # fill rec_aidx_seb with random entries from CF/CB until NO_RECOMMENDED_ARTISTS can be returned
                while(len(rec_aidx_seb) < NO_RECOMMENDED_ARTISTS):
                    if(i%2==0):
                        key = random.choice(rec_aidx_CF.keys())
                        rec_aidx_CF.pop(key)
                    else:
                        key = random.choice(rec_aidx_CB.keys())
                        rec_aidx_CB.pop(key)

                    rec_aidx_seb = np.append(rec_aidx_seb, key)
                    np.unique(rec_aidx_seb)
                    i+=1

                dict_rec_aidx = {}

                for aidx in rec_aidx_seb:
                    dict_rec_aidx[aidx] = 1.0

            elif METHOD == "HR_SCB":
                dict_rec_aidx_CB = recommend_CB(AAM, u_aidx[train_aidx], K_CB, NO_RECOMMENDED_ARTISTS)
                dict_rec_aidx_CF = recommend_CF(UAM.copy(), u, u_aidx[train_aidx], K_CF)
                # Fuse scores given by CF and by CB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(2, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CB.keys():
                    scores[0, aidx] = dict_rec_aidx_CB[aidx]
                for aidx in dict_rec_aidx_CF.keys():
                    scores[1, aidx] = dict_rec_aidx_CF[aidx]
                # Apply aggregation function (here, just take arithmetic mean of scores)
                scores_fused = np.mean(scores, axis=0)
                # Sort and select top K_HR artists to recommend
                sorted_idx = np.argsort(scores_fused)
                sorted_idx_top = sorted_idx[-1 - K_HR:]
                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]
            
            elif METHOD == "DF_GENDER":
                dict_rec_aidx = Recommender_CFDF.recommender_cfdf_gender(u, copy_UAM, NO_RECOMMENDED_ARTISTS, K_CF, USERS_EXTENDED)
            
            elif METHOD == "DF_AGE":
                dict_rec_aidx = Recommender_CFDF.recommender_cfdf_age(u, copy_UAM, NO_RECOMMENDED_ARTISTS, K_CF, USERS_EXTENDED)

            elif METHOD == "DF_COUNTRY":
                dict_rec_aidx = Recommender_CFDF.recommender_cfdf_country(u, copy_UAM, NO_RECOMMENDED_ARTISTS, K_CF, USERS_EXTENDED)


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
    UAM = pd.read_csv(UAM_FILE, delimiter='\t', dtype=np.float32, header=None).values # greatly increase reading speed via pandas
    print "Done."
    # Load AAM
    print "Loading AAM... ",
    #AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)
    # AAM = pd.read_csv(AAM_FILE, delimiter='\t', dtype=np.float32, header=None).values # greatly increase reading speed via pandas
    print "Done."
    print "Loading USERS_EXTENDED...",
    USERS_EXTENDED = pd.read_csv(USERS_EXTENDED_FILE, delimiter='\t', header=0).values
    print "Done."

    if False:
        METHOD = "RB"
        print METHOD
        K_RB = NO_RECOMMENDED_ARTISTS
        print (str(K_RB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1: MAP: 0.36, MAR: 0.01, F1 Score: 0.02
        # NO_RECOMMENDED_ARTISTS = 5: MAP: 0.36, MAR: 0.06, F1 Score: 0.10
        # NO_RECOMMENDED_ARTISTS = 10: MAP: 0.35, MAR: 0.50, F1 Score: 0.41
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 0.35, MAR: 0.20, F1 Score: 0.26
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 0.37, MAR: 0.11, F1 Score: 0.16
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 0.37, MAR: 0.78, F1 Score: 0.50
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 0.37, MAR: 1.03, F1 Score: 0.54
        # NO_RECOMMENDED_ARTISTS = 200: MAP: 0.32, MAR: 1.99, F1 Score: 0.55
        # NO_RECOMMENDED_ARTISTS = 300: MAP: 0.32, MAR: 3.03, F1 Score: 0.58

    if False:
        METHOD = "RB_U"
        print METHOD
        K_RB = NO_RECOMMENDED_ARTISTS
        print (str(K_RB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1: MAP: 0.96, MAR: 0.04, F1 Score: 0.07
        # NO_RECOMMENDED_ARTISTS = 5: MAP: 1.01, MAR: 0.18, F1 Score: 0.30
        # NO_RECOMMENDED_ARTISTS = 10: MAP: 0.98, MAR: 0.33, F1 Score: 0.50
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 0.95, MAR: 0.70, F1 Score: 0.81
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 0.99, MAR: 1.73, F1 Score: 1.26
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 1.02, MAR: 2.73, F1 Score: 1.48
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 1.00, MAR: 3.51, F1 Score: 1.56
        # NO_RECOMMENDED_ARTISTS = 200: MAP: 0.96, MAR: 6.57, F1 Score: 1.68
        # NO_RECOMMENDED_ARTISTS = 300: MAP: 0.92, MAR: 9.40, F1 Score: 1.67

    if False:
        METHOD = "CF"
        print METHOD
        K_CF = 25
        print (str(K_CF) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1: MAP: MAP: 15.98, MAR: 0.70, F1 Score: 1.34
        # NO_RECOMMENDED_ARTISTS = 5: MAP: MAP: 13.16, MAR: 2.75, F1 Score: 4.56
        # NO_RECOMMENDED_ARTISTS = 10: MAP: MAP: 11.63, MAR: 4.79, F1 Score: 6.78
        # NO_RECOMMENDED_ARTISTS = 20: MAP: MAP: 9.81, MAR: 7.89, F1 Score: 8.75
        # NO_RECOMMENDED_ARTISTS = 50: MAP: MAP: 7.38, MAR: 14.38, F1 Score: 9.75
        # NO_RECOMMENDED_ARTISTS = 75: MAP: MAP: 6.36, MAR: 18.31, F1 Score: 9.44
        # NO_RECOMMENDED_ARTISTS = 100: MAP: MAP: 5.66, MAR: 21.55, F1 Score: 8.97
        # NO_RECOMMENDED_ARTISTS = 200: MAP: MAP: 4.14, MAR: 30.94, F1 Score: 7.31
        # NO_RECOMMENDED_ARTISTS = 300: MAP: MAP: 3.38, MAR: 37.40, F1 Score: 6.20

    if False:
        METHOD = "CB"
        print METHOD
        K_CB = NO_RECOMMENDED_ARTISTS
        print (str(K_CB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1: MAP: 4.07, MAR: 1.56, F1 Score: 2.26
        # NO_RECOMMENDED_ARTISTS = 5: MAP: 3.75, MAR: 0.85, F1 Score: 1.39
        # NO_RECOMMENDED_ARTISTS = 10: MAP: 3.37, MAR: 1.45, F1 Score: 2.02
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 2.92, MAR: 2.44, F1 Score: 2.66
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 2.30, MAR: 4.72, F1 Score: 3.09
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 2.01, MAR: 6.07, F1 Score: 3.02
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 1.85, MAR: 7.34, F1 Score: 2.95
        # NO_RECOMMENDED_ARTISTS = 200: MAP: 1.34, MAR: 12.47, F1 Score: 2.42
        # NO_RECOMMENDED_ARTISTS = 300: MAP: 1.15, MAR: 15.61, F1 Score: 2.15


    if False:
        METHOD = "HR_BRB"
        print METHOD
        K_CB = NO_RECOMMENDED_ARTISTS     # number of nearest neighbors to consider in CB (= artists)
        K_CF = 25                         # number of nearest neighbors to consider in CF (= users)
        print (str(K_CB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1:
        # NO_RECOMMENDED_ARTISTS = 5:
        # NO_RECOMMENDED_ARTISTS = 10:
        # NO_RECOMMENDED_ARTISTS = 20:
        # NO_RECOMMENDED_ARTISTS = 50:
        # NO_RECOMMENDED_ARTISTS = 75:
        # NO_RECOMMENDED_ARTISTS = 100:
        # NO_RECOMMENDED_ARTISTS = 200:
        # NO_RECOMMENDED_ARTISTS = 300:

    if False:
        METHOD = "PB"
        print METHOD
        K_PB = NO_RECOMMENDED_ARTISTS
        print (str(K_PB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1:
        # NO_RECOMMENDED_ARTISTS = 5:
        # NO_RECOMMENDED_ARTISTS = 10:
        # NO_RECOMMENDED_ARTISTS = 20:
        # NO_RECOMMENDED_ARTISTS = 50:
        # NO_RECOMMENDED_ARTISTS = 75:
        # NO_RECOMMENDED_ARTISTS = 100:
        # NO_RECOMMENDED_ARTISTS = 200:
        # NO_RECOMMENDED_ARTISTS = 300:

    if False:
        METHOD = "HR_SEB"
        print METHOD
        K_CF = 25
        K_CB = NO_RECOMMENDED_ARTISTS
        print (str(K_CB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1:
        # NO_RECOMMENDED_ARTISTS = 5:
        # NO_RECOMMENDED_ARTISTS = 10:
        # NO_RECOMMENDED_ARTISTS = 20:
        # NO_RECOMMENDED_ARTISTS = 50:
        # NO_RECOMMENDED_ARTISTS = 75:
        # NO_RECOMMENDED_ARTISTS = 100:
        # NO_RECOMMENDED_ARTISTS = 200:
        # NO_RECOMMENDED_ARTISTS = 300:

    if False:
        METHOD = "HR_SCB"
        print METHOD
        K_CB = NO_RECOMMENDED_ARTISTS
        K_HR = NO_RECOMMENDED_ARTISTS
        K_CF = 25
        print (str(K_CB) + ","),
        run()
        # NO_RECOMMENDED_ARTISTS = 1: MAP: 9.66, MAR: 0.78, F1 Score: 1.45
        # NO_RECOMMENDED_ARTISTS = 5: MAP: 8.47, MAR: 2.05, F1 Score: 3.30
        # NO_RECOMMENDED_ARTISTS = 10: MAP: 7.64, MAR: 3.34, F1 Score: 4.64
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 6.74, MAR: 5.57, F1 Score: 6.10
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 5.44, MAR: 10.82, F1 Score: 7.24
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 4.82, MAR: 14.11, F1 Score: 7.19
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 4.40, MAR: 16.98, F1 Score: 6.99
        # NO_RECOMMENDED_ARTISTS = 200: MAP: 3.41, MAR: 25.76, F1 Score: 6.03
        # NO_RECOMMENDED_ARTISTS = 300: MAP: 2.89, MAR: 32.26, F1 Score: 5.30

    if False:
        METHOD = "DF_GENDER"
        print METHOD
        K_CF = 25
        run()
        # NO_RECOMMENDED_ARTISTS = 1: MAP: 14.66, MAR: 0.63, F1 Score: 1.21
        # NO_RECOMMENDED_ARTISTS = 5: MAP: 12.30, MAR: 2.57, F1 Score: 4.26
        # NO_RECOMMENDED_ARTISTS = 10: MAP: 10.98, MAR: 4.51, F1 Score: 6.40
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 9.32, MAR: 7.47, F1 Score: 8.29
        # NO_RECOMMENDED_ARTISTS = 50: MAP 7.09, MAR: 13.70, F1 Score: 9.34
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 6.13, MAR: 17.50, F1 Score: 9.08
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 5.49, MAR: 20.65, F1 Score: 8.67
        # NO_RECOMMENDED_ARTISTS = 200: MAP: 4.06, MAR: 29.85, F1 Score: 7.15
        # NO_RECOMMENDED_ARTISTS = 300: MAP: 3.34, MAR: 36.23, F1 Score: 6.12

    if True:
        METHOD = "DF_AGE"
        print METHOD
        K_CF = 25
        run()
        # NO_RECOMMENDED_ARTISTS = 1: MAP: 1.44, MAR: 0.06, F1 Score: 0.11
        # NO_RECOMMENDED_ARTISTS = 5: MAP: 1.42, MAR: 0.28, F1 Score: 0.46
        # NO_RECOMMENDED_ARTISTS = 10: MAP: 1.29, MAR: 0.48, F1 Score: 0.70
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 1.32, MAR: 0.96, F1 Score: 1.11
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 1.31, MAR: 2.44, F1 Score: 1.71
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 1.27, MAR: 3.56, F1 Score: 1.88
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 1.30, MAR: 4.81, F1 Score: 2.05
        # NO_RECOMMENDED_ARTISTS = 200: MAP: 1.23, MAR: 9.09, F1 Score: 2.17
        # NO_RECOMMENDED_ARTISTS = 300: MAP: 1.20, MAR: 13.18, F1 Score: 2.20

    if False:
        METHOD = "DF_COUNTRY"
        print METHOD
        K_CF = 25
        run()
        # NO_RECOMMENDED_ARTISTS = 1: MAP: 12.26, MAR: 0.54, F1 Score: 1.04
        # NO_RECOMMENDED_ARTISTS = 5: MAP: 10.15, MAR: 2.20, F1 Score: 3.61
        # NO_RECOMMENDED_ARTISTS = 10: MAP: 9.04, MAR: 3.90, F1 Score: 5.45
        # NO_RECOMMENDED_ARTISTS = 20: MAP: 7.78, MAR: 6.51, F1 Score: 7.09
        # NO_RECOMMENDED_ARTISTS = 50: MAP: 6.01, MAR: 12.17, F1 Score: 8.05
        # NO_RECOMMENDED_ARTISTS = 75: MAP: 5.24, MAR: 15.69, F1 Score: 7.85
        # NO_RECOMMENDED_ARTISTS = 100: MAP: 4.69, MAR: 18.55, F1 Score: 7.49
        # NO_RECOMMENDED_ARTISTS = 200: MAP: 3.51, MAR: 27.15, F1 Score: 6.22
        # NO_RECOMMENDED_ARTISTS = 300: MAP: 2.90, MAR: 33.12, F1 Score: 5.34