#fetches them by URL with urllib.request or requests
#script is reproducible

"""
Downloads a small corpus of public-domain books from Project Gutenberg 
into the data/ folder, as plain UTF-8 text files.

To run: 
    python src/download.py
"""
import urllib.request
import os

# All three are 19th century English novels, on purpose: keeps the training corpus stylistically consistent.
# Matters for Week 3, where I will test the model on something "different" and see it get worse.

BOOKS = { 1342: "pride_and_prejudice.txt", 84: "frankenstein.txt", 11: "alice_in_wonderland.txt" }

URL_TEMPLATE = "https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
#must call .format() on it

#the data folder is one level up, in data/
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data") 

def download_book(book_id: int, filename: str) -> None:
    url = URL_TEMPLATE.format(book_id = book_id)
    out_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(out_path):
        print(f"Skipping {filename} (already downloaded)")
        return

    print(f"Donwloading {filename} from {url} ...")
    #urlretrieve fetches the URL and writes the response body straight to disk
    urllib.request.urlretrieve(url, out_path)
    print(f"    saved to {out_path}")


def main():
    os.makedirs(DATA_DIR, exist_ok = True)
    for book_id, filename in BOOKS.items():
        download_book(book_id, filename)

if __name__ == "__main__":
    main()