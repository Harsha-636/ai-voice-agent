# 🎙️ Personal AI Voice Agent

A voice-enabled AI assistant that answers questions **on your behalf, 24/7**, using only facts from your own knowledge base (RAG) — so answers stay relevant and correct.

## Features
- Voice in/out (browser Web Speech API) + text fallback, English / Hindi / Telugu
- **RAG grounding**: TF-IDF retrieval over `knowledge/*.md`; the LLM may only use retrieved facts
- **Hallucination guard**: unknown questions get an honest "I don't know" + offer to take a message
- **Lead capture**: visitors can leave name/contact/message
- **Admin dashboard** (`/admin`): all conversations, *unanswered questions* to fix, messages
- Rate limiting, prompt-injection-resistant system prompt, SQLite logging
- **Hybrid retrieval** (BM25 + typo-tolerant trigram match + synonyms), no heavy ML dependencies
- **Phone line**: Twilio number → `/api/phone/voice` (speech-to-speech calls, logged as channel `phone`)
- **Instant alerts** for new messages via `NOTIFY_WEBHOOK` (Discord/Slack webhook)
- **Tests + CI**: pytest suite with a retrieval accuracy check, GitHub Actions, Dockerfile
- Works without an API key in demo (retrieval-only) mode

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in values, then: export $(cat .env | xargs)
uvicorn app:app --reload
```
Open http://localhost:8000 (admin: /admin).

## Deploy free (Render)
Push to GitHub → Render → New Web Service → pick repo (it reads `render.yaml`) → set `ANTHROPIC_API_KEY` and `ADMIN_TOKEN`.
Note: free tier sleeps when idle and SQLite resets on redeploy.

## Architecture
Browser (speech→text) → FastAPI `/api/chat` → retrieve top chunks → Claude with grounded prompt → answer (text→speech)

## Phone setup (optional)
Buy a Twilio number, set its *A call comes in* webhook (POST) to `https://YOUR-APP.onrender.com/api/phone/voice?s=SECRET`, and set `PHONE_SECRET=SECRET` in Render.

## Tests
`pip install -r requirements-dev.txt && pytest -q -s`

## Roadmap
Embedding retrieval, streaming replies, persistent Postgres, Twilio signature validation.
