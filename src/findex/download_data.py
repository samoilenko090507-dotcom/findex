import urllib.request
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

BOOKS = {
    "frankenstein.txt": "https://www.gutenberg.org/files/84/84-0.txt",
    "dracula.txt": "https://www.gutenberg.org/files/345/345-0.txt",
    "alice.txt": "https://www.gutenberg.org/files/11/11-0.txt",
    "sherlock.txt": "https://www.gutenberg.org/files/1661/1661-0.txt",
}

for name, url in BOOKS.items():
    file_path = DATA_DIR / name
    if not file_path.exists():
        print(f"Завантаження {name}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(file_path, "wb") as f:
                f.write(resp.read())
        except Exception as e:  # noqa: BLE001
            print(f"Помилка завантаження {name}: {e}")

print("Усі доступні тексти завантажено в папку data/")
