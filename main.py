"""
Phase 0 deliverable: a minimal FastAPI app.
Run with: uvicorn main:app --reload
Then visit http://127.0.0.1:8000 in your browser.
"""

from fastapi import FastAPI
from dotenv import load_dotenv

# Loads variables from your .env file (like ANTHROPIC_API_KEY)
# so they're accessible via os.environ later.
load_dotenv()

app = FastAPI(title="Ask My Docs")


@app.get("/")
def health_check():
    return {"status": "ok"}
