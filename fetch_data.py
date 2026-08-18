"""
Phase 1, Step A: fetch source documents.

This script downloads a handful of Wikipedia articles about banking and
financial history and saves each one as a plain .txt file inside data/.
These become the raw documents our RAG system will later chunk, embed,
and search over.

Uses Wikipedia's official API directly (more reliable than third-party
wrapper packages, which often break).

Run with: python fetch_data.py
"""

import requests
import os

ARTICLE_TITLES = [
    "History of banking",
    "Federal Reserve",
    "Great Depression",
    "Bretton Woods system",
    "2007-2008 financial crisis",
]

OUTPUT_DIR = "data"
API_URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia requires a descriptive User-Agent identifying the request source.
HEADERS = {
    "User-Agent": "AskMyDocsLearningProject/1.0 (educational RAG project)"
}


def fetch_and_save(title: str):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,  # plain text, not HTML
        "redirects": 1,
    }

    response = requests.get(API_URL, params=params, headers=HEADERS)
    response.raise_for_status()  # raises an error if the request failed
    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))  # get the single page object

    if "extract" not in page or not page["extract"]:
        print(f"WARNING: no content found for '{title}', skipping.")
        return

    real_title = page["title"]
    content = page["extract"]

    safe_name = real_title.replace(" ", "_").replace("/", "-")
    filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved: {filepath} ({len(content)} characters)")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for title in ARTICLE_TITLES:
        fetch_and_save(title)
    print("\nDone! Check your data/ folder.")
