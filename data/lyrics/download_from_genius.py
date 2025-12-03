import lyricsgenius

with open("token.txt", "r") as f:
    token = f.read().strip()
genius = lyricsgenius.Genius(token)

genius.remove_section_headers = True 
genius.timeout = 20
genius.retries = 5

for nazwa_artysty in ["white2115", "hałastra", "young igi", "young multi", "catchup"]:
    try:
        filename = f"Teksty_{nazwa_artysty.replace(' ', '_')}.txt"

        print(f"Szukam artysty: {nazwa_artysty}...")
        
        artist = genius.search_artist(nazwa_artysty, max_songs=None, include_features=False, sort="popularity")

        if artist:
            print(f"Znaleziono artystę: {artist.name}")
            print(f"Rozpoczynam pobieranie {len(artist.songs)} piosenek. To może potrwać...")

            with open(filename, 'w', encoding='utf-8') as f:
                
                for i, song in enumerate(artist.songs):
                    if song and song.lyrics:
                        print (song.primary_artist)
                        print(f"  > Zapisuję [{i+1}/{len(artist.songs)}]: {song.title}")
                        
                        f.write(f"--- {song.title} ---\n\n")
                        f.write(song.lyrics)
                        f.write("\n\n" + "="*40 + "\n\n")
                    else:
                        print(f"  > Pomijam (brak tekstu): {song.title}")

            print(f"\nGotowe! Zapisano teksty do pliku: {filename}")

        else:
            print(f"Nie znaleziono artysty o nazwie: {nazwa_artysty}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        print("Jeśli błąd to 'Timeout', spróbuj jeszcze bardziej zwiększyć wartość 'timeout'.")
