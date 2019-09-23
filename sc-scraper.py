from bs4 import BeautifulSoup
import soundcloud
import requests
import music-dl

class Scraper():
    def scrape_playlist(self, url):
        # soup = BeautifulSoup(url, "lxml")
        r = requests.get(url)
        # print(r.text)
        soup = BeautifulSoup(r.text, 'html.parser')
        print(soup.prettify())

        for link in soup.find_all('article'):
            print(link.a['href'])


# create a client object with access token
client = soundcloud.Client(client_id='q2iUepUBTAabXdJFYY7vjaGn6yno13KB')

# get playlist
playlist = client.get('/playlists/2050462')

# list tracks in playlist
for track in playlist.tracks:
    print(track['stream_url'])
    # fetch track to stream
    # get the tracks streaming URL
    stream_url = client.get(track['stream_url'], allow_redirects=False)

    # print the tracks stream URL
    print(stream_url.location)


if __name__ == '__main__':
    s = Scraper()
    s.scrape_playlist("https://soundcloud.com/delfino666/sets/training-02")
