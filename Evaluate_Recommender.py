# Load required modules
import csv
import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
from Simple_Recommender_CF import simple_recommender_cf

# Parameters
UAM_FILE = "./data/C1ku_UAM.txt"                    # user-artist-matrix (UAM)
ARTISTS_FILE = "./data/C1ku_idx_artists.txt"        # artist names for UAM
USERS_FILE = "./data/C1ku_idx_users.txt"            # user names for UAM

NF = 10              # number of folds to perform in cross-validation TODO 10 folds

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

def create_training_UAM(UAM, seed_uidx, seed_aidx_train):
    # UAM               user-artist-matrix. Is changed by this function!
    # seed_uidx         user index of seed user
    # seed_aidx_train   indices in UAM of training artists for seed user

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Remove information on test artists from seed's listening vector
    aidx_nz = np.nonzero(pc_vec)[0]                            # artists with non-zero listening events
    aidx_test = np.setdiff1d(aidx_nz, seed_aidx_train)         # compute set difference between all artist indices of user and train indices gives test artist indices

    # Set to 0 the listening events of seed user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx, :] = UAM[seed_uidx, :] / np.sum(UAM[seed_uidx, :])


# Function that implements dumb random recommender. It predicts a number of randomly chosen items.
# It returns a list of recommended artist indices.
def recommend_RB(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Return list of recommended artist indices
    return random_aidx


# Function that implements a baseline recommender. It uses the artists of a randomly chosen user as predictions.
# It returns a list of recommended artist indices.
def recommend_artists_from_random_user(u_aidx, all_other_users_idx, UAM, no_items):
    # u_aidx                 user artist indices
    # all_other_users_idx    indexes from where we choose a random user
    # no_items               no of items to predict

    # Let's predict a random user
    random_user_idx = random.sample(all_other_users_idx, 1)

    random_user_artists_idx = np.nonzero(UAM[random_user_idx, :])[1]

    random_user_artists_idx = np.setdiff1d(random_user_artists_idx, u_aidx)

    # Get artists where number is no_items
    if len(random_user_artists_idx) < no_items:
        no_items = len(random_user_artists_idx)
    random_user_artists_idx = random.sample(random_user_artists_idx, no_items)

    # Return list of recommended artist indices
    return random_user_artists_idx


# Main program
if __name__ == '__main__':

    # Initialize variables to hold performance measures
    avg_prec = 0;       # mean precision
    avg_rec = 0;        # mean recall

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    # For all users in our data (UAM)
    no_users = UAM.shape[0]

    # Count number of users with spare data (users which have less than NF unique artists)
    no_sparse_users = 0
    for user in range(0, no_users):
        u_aidx = np.nonzero(UAM[user, :])[0]
        if len(u_aidx) < NF:
            no_sparse_users += 1

    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]
        print "nr. of artists: " + str(len(u_aidx))

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0

        if len(u_aidx) < NF:
            # ignore sparse users
            continue
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV
        for train_aidx, test_aidx in kf:  # for all folds
            # Show progress
            print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                len(train_aidx)) + ", Test items: " + str(len(test_aidx)),      # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()       # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable
            #rec_aidx = recommend_CF(copy_UAM, u, u_aidx[train_aidx])
            create_training_UAM(copy_UAM, u, train_aidx)
            # rec_aidx = simple_recommender_cf(u, copy_UAM, len(test_aidx), 1)  ###############
            # rec_aidx = simple_recommender_cf(u, copy_UAM, len(test_aidx), 2)  ###############
            # rec_aidx = simple_recommender_cf(u, copy_UAM, len(test_aidx), 3)  ###############
            # rec_aidx = simple_recommender_cf(u, copy_UAM, len(test_aidx), 5)  ###############
            # rec_aidx = simple_recommender_cf(u, copy_UAM, len(test_aidx), 10)  ###############
            # rec_aidx = simple_recommender_cf(u, copy_UAM, len(test_aidx), 20)  ###############
#            print "Recommended items: ", len(rec_aidx)

            # For random recommendation, exclude items that the user already knows, i.e. the ones in the training set
            # all_aidx = range(0, UAM.shape[1])
            # rec_aidx = recommend_RB(np.setdiff1d(all_aidx, u_aidx[train_aidx]), len(test_aidx))       # select the number of recommended items as the number of items in the test set

            # Our baseline:
            all_other_users_idx = range(no_users)
            all_other_users_idx = np.setdiff1d(all_other_users_idx, u)
            # rec_aidx = recommend_artists_from_random_user(train_aidx, all_other_users_idx, copy_UAM, 1) # MAP: 0.80, MAR: 0.03, F1 Score: 0.06
            # rec_aidx = recommend_artists_from_random_user(train_aidx, all_other_users_idx, copy_UAM, 5) # MAP: 0.68, MAR: 0.12, F1 Score: 0.21
            # rec_aidx = recommend_artists_from_random_user(train_aidx, all_other_users_idx, copy_UAM, 10) # MAP: 0.68, MAR: 0.26, F1 Score: 0.37
            # rec_aidx = recommend_artists_from_random_user(train_aidx, all_other_users_idx, copy_UAM, 20) # MAP: 0.67, MAR: 0.51, F1 Score: 0.58
            # rec_aidx = recommend_artists_from_random_user(train_aidx, all_other_users_idx, copy_UAM, 50) # MAP: 0.66, MAR: 1.19, F1 Score: 0.85
            # rec_aidx = recommend_artists_from_random_user(train_aidx, all_other_users_idx, copy_UAM, 100) # MAP: 0.69, MAR: 2.18, F1 Score: 1.05


            print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)          # correctly predicted artists
#            print 'Recommended artist-ids: ', rec_aidx
#            print 'True artist-ids: ', u_aidx[test_aidx]

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
            avg_prec += prec / (NF * (no_users - no_sparse_users))
            avg_rec += rec / (NF * (no_users - no_sparse_users))

            # Output precision and recall of current fold
            print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

            # Increase fold counter
            fold += 1

    # Output mean average precision and recall
    f1_score = 2 * ( (avg_prec * avg_rec) / (avg_prec + avg_rec))

    print ("\nMAP: %.2f, MAR: %.2f, F1 Score: %.2f" % (avg_prec, avg_rec, f1_score))
