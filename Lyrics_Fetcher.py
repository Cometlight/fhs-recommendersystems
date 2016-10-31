import urllib
from bs4 import BeautifulSoup
import requests
import os
import Helper_IO as io
from joblib import Parallel, delayed

LASTFM_ARTIST_URL = "http://www.last.fm/music/{}/+tracks" # for fetching of top songs. {} -> artist
WIKI_ARTIST_URL = "http://lyrics.wikia.com/wiki/{}:{}" # for fetching of lyrics of songs. {} -> artist, {} -> song name
NUMBER_OF_SONGS_PER_ARTIST = 10

ARTISTS_FILE = "./data/C1ku_artists.csv"
OUTPUT_DIRECTORY = "./data/crawls_lyrics/"

def get_top_songs(artist, count):
    # artist .. the artist's name as a string
    # count .. how many songs to return maximally
    # return .. list of top songs as strings, or empty if none found
    url = LASTFM_ARTIST_URL.format(urllib.quote(artist))
    request = requests.get(url)

    top_tracks_names = []
    soup = BeautifulSoup(request.text, "html.parser")
    try:
        top_tracks_section = soup.find("section", {"id": "artist-tracks-section"})
        top_tracks_table = top_tracks_section.findChildren("table")[0]
        top_tracks_td_name = top_tracks_table.findChildren("td", {"class": "chartlist-name"}, limit=count)
        for td_name in top_tracks_td_name:
            name = td_name.findChildren("a", {"class": "link-block-target"})[0].text
            top_tracks_names.append(name.encode("utf-8"))
    except (AttributeError, IndexError) as e:
        print "No top songs found for artist '" + artist + "'"

    return top_tracks_names

def get_lyrics(artist, song):
    # artist .. the artist's name as a string
    # song .. the song's name as a string
    # return .. the song's lyrics, or empty string if not found
    url = WIKI_ARTIST_URL.format(urllib.quote(artist), urllib.quote(song))
    request = requests.get(url)

    lyrics = ""
    soup = BeautifulSoup(request.text, "html.parser")
    try:
        lyricbox = soup.find("div", {"class": "lyricbox"})
        lyrics = u' '.join(lyricbox.findAll(text=True)) # if we'd use lyricbox.text instead, we'd loose all the whitespaces
    except AttributeError as e:
        print "No lyrics found for " + artist + "'s song '" + song + "'"
    
    return lyrics.encode("utf-8")

def fetch_and_save_lyrics_for_artist(artist_id, artists):
    print "Fetching lyrics of artist {} ({} artists in total)".format(artist_id, len(artists))
    file_name = OUTPUT_DIRECTORY + str(artist_id) + ".txt"

    top_songs = get_top_songs(artists[artist_id], NUMBER_OF_SONGS_PER_ARTIST)
    lyrics_list = [get_lyrics(artists[artist_id], song_name) for song_name in top_songs]
    lyrics_list = filter(None, lyrics_list) # Remove empty lyrics for which no text could be found
    lyrics_combined = " ".join(lyrics_list)

    with open(file_name, 'w') as f:
        f.write(lyrics_combined)

if __name__ == '__main__':
    # Create output directory if non-existent
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Read artist list
    artists = io.read_file(ARTISTS_FILE)

    # Let's fetch the artists' lyrics
    num_parallel_jobs = 8
    Parallel(n_jobs=num_parallel_jobs)(delayed(fetch_and_save_lyrics_for_artist)(i,artists) for i in range(2510, len(artists)))
    
    # Original for-loop implementation, before joblib was used:
    # for i in range(0, len(artists)):
        # print "Fetching lyrics of artist {} of {}".format(i, len(artists))
        # file_name = OUTPUT_DIRECTORY + str(i) + ".txt"

        # top_songs = get_top_songs(artists[i], NUMBER_OF_SONGS_PER_ARTIST)
        # lyrics_list = [get_lyrics(artists[i], song_name) for song_name in top_songs]
        # lyrics_list = filter(None, lyrics_list) # Remove empty lyrics for which no text could be found
        # lyrics_combined = " ".join(lyrics_list)

        # with open(file_name, 'w') as f:
        #     f.write(lyrics_combined)

    print "Fetched and saved all lyrics."
