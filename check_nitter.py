# Create this as check_nitter.py in your project root
import requests

headers = {"User-Agent": "Mozilla/5.0"}

instances = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.woodland.cafe",
]

search_term = "drug+shortage"

for base in instances:
    url = f"{base}/search/rss?f=tweets&q={search_term}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        has_rss = "<rss" in r.text[:500] or "<item" in r.text[:1000]
        items = r.text.count("<item")
        print(f"{r.status_code} items={items:3d} {base}")
    except Exception as e:
        print(f"FAIL        {base} ({type(e).__name__})")
