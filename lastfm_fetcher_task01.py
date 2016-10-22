# Load required modules
import os
import urllib
import csv
import json


# Parameters
LASTFM_API_KEY = "af642fd2ce0b0e8eb99519a187657101"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"
LASTFM_OUTPUT_FORMAT = "json"

SEED_USERS_FILE = "./data/lastfm_users_5.csv"
OUTPUT_USERS_DIRTY_FILE = "./data/users_dirty.json"
OUTPUT_LISTENING_EVENTS = "./data/listening_events.csv"

MAX_PAGES = 5                   # maximum number of pages per user
MAX_EVENTS_PER_PAGE = 200       # maximum number of listening events to retrieve per page

PLAYCOUNT_MINIMUM = 1000

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

def filter_users(users_dirty):
    filtered_users = []
    for user_dirty in users_dirty:
        if user_dirty["playcount"] > PLAYCOUNT_MINIMUM:
            filtered_users.append(user_dirty)
    return filtered_users

def get_listening_events(username):
    listening_events = []

    for p in range(0, MAX_PAGES):
        url = construct_api_call('?method=user.getrecenttracks' + \
            '&limit=' + str(MAX_EVENTS_PER_PAGE) + \
            '&page=' + str(p+1) + \
            '&user=' + urllib.quote(username))
        response = urllib.urlopen(url).read()
        json_response = json.loads(response)
        for track in json_response["recenttracks"]["track"]:
            listening_event = []
            listening_event.append(username)
            listening_event.append(track["artist"]["mbid"])
            listening_event.append(track["artist"]["#text"])
            listening_event.append(track["mbid"])
            listening_event.append(track["name"])
            listening_event.append(track["date"]["uts"])

            listening_events.append(listening_event)
    return listening_events



# Main program
if __name__ == '__main__':
    seed_users = read_users(SEED_USERS_FILE)
    users_dirty = []
    for user in seed_users:
        friends = get_friends(user)
        users_dirty += friends

    # TODO: Refactor: Filter everything at the end
    filtered_users = filter_users(users_dirty)
    
    with open(OUTPUT_USERS_DIRTY_FILE, 'w') as f:
        f.write(json.dumps(filtered_users))

    listening_events = []
    for json_user in filtered_users:
        listening_events += get_listening_events(json_user["name"])
    
    with open(OUTPUT_LISTENING_EVENTS, 'w') as f:
       writer = csv.writer(f, delimiter=',')
       writer.writerows(listening_events)