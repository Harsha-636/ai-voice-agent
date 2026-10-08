import os, json, tempfile, pathlib
os.environ.update(KNOWLEDGE_DIR=str(pathlib.Path(__file__).parent / "sample_knowledge"), DB_PATH=tempfile.mktemp(), ADMIN_TOKEN="t0k", ANTHROPIC_API_KEY="")
os.environ.pop("ANTHROPIC_API_KEY")
from fastapi.testclient import TestClient
import app
c = TestClient(app.app)

def test_retrieval_eval():
    gold = json.load(open(pathlib.Path(__file__).parent / "golden.json"))
    hit = sum(bool(app.retrieve(q)) and app.retrieve(q)[0][2].startswith("# " + h) for q, h in gold)
    print(f"retrieval top-1 accuracy: {hit}/{len(gold)}")
    assert hit / len(gold) >= 0.85

def test_unknown_question_is_honest():
    r = c.post("/api/chat", json={"message": "what is your favourite pizza zebra"}).json()
    assert r["known"] is False

def test_admin_auth_and_stats():
    assert c.get("/api/admin/data", headers={"x-token": "bad"}).status_code == 401
    d = c.get("/api/admin/data", headers={"x-token": "t0k"}).json()
    assert d["stats"]["total"] >= 1

def test_phone_twiml():
    assert "<Gather" in c.post("/api/phone/voice").text
    r = c.post("/api/phone/answer", content="SpeechResult=what+are+your+skills", headers={"content-type": "application/x-www-form-urlencoded"})
    assert "Python" in r.text

def test_empty_and_message():
    assert c.post("/api/chat", json={"message": "  "}).status_code == 400
    assert c.post("/api/message", json={"name": "a", "contact": "b", "body": "hi"}).json()["ok"]
