import lyricsgenius as genius
import json
import os

api = genius.Genius('4eMJoKWuTAyfxabfE_CZekpjNgp1JBTD86ULdSC3JfUGmHnLrz853VKif2pLMZdE')

artist_name = 'KendrickLamar'
artist = api.search_artist(artist_name,max_songs=200)

artist.save_lyrics()

with open('Lyrics_{}.json'.format(artist_name)) as json_data:
    data = json.load(json_data)
os.remove('./Lyrics_{}.json'.format(artist_name))

lyrics=[]
for song in data['songs']:
    lyrics.append(song['lyrics'])

filepath = "./data/"+artist_name+"_Lyrics.txt"

f = open(filepath,'w')
for song in lyrics:
    f.write(str(song))
f.close()