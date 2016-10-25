# Load required modules
import os
import urllib
import csv
import json
import numpy as np


# Parameters
LASTFM_API_KEY = "af642fd2ce0b0e8eb99519a187657101"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"
LASTFM_OUTPUT_FORMAT = "json"

SEED_USERS_FILE = "./data/lastfm_users_100.csv"
OUTPUT_USERS_DIRTY_FILE = "./data/users_dirty.json"
OUTPUT_RAW_LISTENING_EVENTS = "./data/listening_events_raw.csv"
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

# Build API call based on method & params (str)
def construct_api_call(str):
    return LASTFM_API_URL + str + '&api_key=' + LASTFM_API_KEY + '&format=json'

# Get friends of a specific user by username
def get_friends(username):
    url = construct_api_call('?method=user.getfriends&user='+urllib.quote(username))
    response = urllib.urlopen(url).read()
    json_users = json.loads(response)

    if not "friends" in json_users:
        return []

    return json_users["friends"]["user"]

# Remove users with a playcount smaller than PLAYCOUNT_MINIMUM
def filter_users(users_dirty):
    filtered_users = []
    for user_dirty in users_dirty:
        if int(user_dirty["playcount"]) > PLAYCOUNT_MINIMUM:
            filtered_users.append(user_dirty)
    return filtered_users

# Get listening events for a specific user by username
def get_listening_events(username):
    listening_events = []

    for p in range(0, MAX_PAGES):
        url = construct_api_call('?method=user.getrecenttracks' + \
            '&limit=' + str(MAX_EVENTS_PER_PAGE) + \
            '&page=' + str(p+1) + \
            '&user=' + urllib.quote(username))
        response = urllib.urlopen(url).read()
        json_response = json.loads(response)
        try:
            for track in json_response["recenttracks"]["track"]:
                # Data cleansing: All tracks are removed which ..
                #   .. have no date
                #   .. have no track musicbrainz id
                #   .. have no artist
                #   .. have no artist musicbrainz id
                if (not "date" in track or track["date"] == "" \
                    or not "mbid" in track or track["mbid"] == "" \
                    or not "artist" in track or track["artist"] == "" \
                    or not "mbid" in track["artist"] or track["artist"]["mbid"] == ""):
                    continue
                listening_event = []
                listening_event.append(username)
                # listening_event.append(track["artist"]["mbid"])
                listening_event.append(track["artist"]["#text"])
                # listening_event.append(track["mbid"])
                listening_event.append(track["name"])
                listening_event.append(track["date"]["uts"])

                listening_events.append(listening_event)
        except:
            print "Unexpected Error"
            continue

    return listening_events

# Main program
if __name__ == '__main__':

    seed_users = read_users(SEED_USERS_FILE)
    users_dirty = []

    # get friends of all seed users
    for user in seed_users:
        friends = get_friends(user)
        users_dirty += friends

    # fetch friends of already fetched users until we have at least 2500 users
    i = 0
    while len(users_dirty) < 2500:
        i+=1
        friends = get_friends(users_dirty[i]["name"])
        users_dirty += friends

    filtered_users = filter_users(users_dirty)
    
    with open(OUTPUT_USERS_DIRTY_FILE, 'w') as f:
        f.write(json.dumps(filtered_users))

    listening_events = []
    cleansed_users = []
    artists_count = {}
    cleansed_artists = []
    cleansed_listening_events = []

    # get listening events for each user
    for json_user in filtered_users:
        listening_events_user = get_listening_events(json_user["name"])

        try:
            leu = np.array( listening_events_user )
            artists = np.unique(leu[:,1])

            # user cleansing: add users with more than 10 unique artists
            if len(artists) > 10:
                listening_events += listening_events_user
                cleansed_users.append(json_user)

            # count artists (for cleansing)
            for artist in artists:
                if artist in artists_count:
                    artists_count[artist] += 1
                else:
                    artists_count[artist] = 1
        except IndexError:
            print "IndexError:"
            print leu
            continue
        except:
            print "UnexpectedError"
            continue

    # save raw listening events
    with open(OUTPUT_RAW_LISTENING_EVENTS, 'w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        for listening_event in listening_events:
            writer.writerow([unicode(element).encode("utf-8") for element in listening_event])

    # START CLEANSING
    # remove artists with less than 10 unique users
    for artist, count in artists_count.iteritems():
        if count > 10:
            cleansed_artists.append(artist)

    # remove listening items with removed artists
    for le in listening_events:
        if le[1] in cleansed_artists:
            cleansed_listening_events.append(le)

    # save cleansed listening events
    with open(OUTPUT_LISTENING_EVENTS, 'w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        for listening_event in cleansed_listening_events:
            writer.writerow([unicode(element).encode("utf-8") for element in listening_event])
