# Load required modules
import os
import urllib
import csv
import json


# Parameters
LASTFM_API_KEY = "af642fd2ce0b0e8eb99519a187657101"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"
LASTFM_OUTPUT_FORMAT = "json"

SEED_USERS_FILE = "./data/lastfm_users_100.csv"
OUTPUT_USERS_DIRTY_FILE = "./data/users_dirty.json"

# Simple function to read content of a text file into a list
def read_users(users_file):
    users = []                                      # list to hold user names
    with open(users_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')      # create reader
#        headers = reader.next()                    # in case we have a header
        for row in reader:
            users.append(row[0])
    return users

def construct_api_call(str):
    return LASTFM_API_URL + str + '&api_key=' + LASTFM_API_KEY + '&format=json'


def get_friends(username):
    url = construct_api_call('?method=user.getfriends&user='+urllib.quote(username))
    response = urllib.urlopen(url).read()
    json_users = json.loads(response)

    if not "friends" in json_users:
        return []

    return json_users["friends"]["user"]

# Main program
if __name__ == '__main__':
    seed_users = read_users(SEED_USERS_FILE)
    users_dirty = []
    for user in seed_users:
        friends = get_friends(user)
        users_dirty += friends
    
    with open(OUTPUT_USERS_DIRTY_FILE, 'w') as f:
        f.write(json.dumps(users_dirty))
        
