"""Page navigation, isolation, and truthful availability checks."""

from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.categories import CATEGORIES
from web.lessons import LESSONS

client = TestClient(app)


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.current = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href", "").startswith("/"):
            self.links.append(attrs["href"])
        if attrs.get("aria-current") == "page":
            self.current.append(attrs.get("href"))


@pytest.mark.parametrize("path,active", [("/", "/"), ("/practice", "/practice"), ("/writing", "/writing")] + [
    (f"/practice/{category_id}", "/practice") for category_id in CATEGORIES
])
def test_pages_and_all_internal_links(path, active):
    response = client.get(path)
    assert response.status_code == 200
    assert 'lang="zh-Hant"' in response.text
    parser = PageLinks()
    parser.feed(response.text)
    assert parser.current == [active]
    for link in parser.links:
        assert client.get(link).status_code == 200, (path, link)


def test_all_seven_topics_visible_including_degree_adverb():
    html = client.get("/practice").text
    assert len(CATEGORIES) == 7
    for category_id, category in CATEGORIES.items():
        assert f'data-category="{category_id}"' in html
        assert category["name"] in html
        assert category["english"] in html
    assert 'href="/practice/degree_adverb_misuse"' in html


def test_empty_dashboard_does_not_invent_progress():
    html = client.get("/").text
    assert 'data-stat="accuracy">-</strong>' in html
    assert 'data-stat="sessions">0</strong>' in html
    assert 'data-stat="xp">0</strong>' in html
    assert "教學例句" in html
    assert 'id="reset-dialog"' in html


def test_readiness_and_writing_do_not_claim_live_models(monkeypatch):
    monkeypatch.setattr(app.state, "mode", "live")
    assert client.get("/health").json() == {
        "web_ready": True, "catalogue_ready": True, "practice_ready": True,
        "model_ready": False, "model_loading": False, "diagnosis_categories": [],
        "mode": "live", "checkpoint": "D", "reviewer": False,
        "sentence_bank_ready": False, "sentence_bank_count": 0,
    }
    html = client.get("/writing").text
    assert "目前尚未連接診斷服務" in html
    assert "不代表完整文法檢查" in html
    assert '<textarea id="sentence-input"' in html
    assert client.post("/api/diagnose", json={"text": "private learner text"}).status_code == 503


@pytest.mark.parametrize("path", [
    "/practice/unknown", "/missing", "/.git/config", "/data/raw/example.xml",
    "/EFCAMDAT_Database.xml", "/schemas/model2-output-v1.schema.json",
    "/static/../app.py", "/static/%2e%2e/app.py", "/static/%2e%2e/%2e%2e/pyproject.toml",
    "/static/questions.json", "/docs/patternlab/PatternLab_v1_Implementation_Spec.md",
])
def test_unknown_routes_and_private_files_are_not_served(path):
    response = client.get(path)
    assert response.status_code == 404
    assert "這一頁暫時找不到" in response.text
    assert "private learner text" not in response.text


@pytest.mark.parametrize("asset,media_type", [("style.css", "text/css"), ("app.js", "javascript"), ("practice.js", "javascript"), ("writing.js", "javascript"), ("favicon.svg", "image/svg+xml"), ("images/pixel-landscape-v1.png", "image/png")])
def test_static_assets(asset, media_type):
    response = client.get(f"/static/{asset}")
    assert response.status_code == 200
    assert media_type in response.headers["content-type"]


def test_every_topic_has_a_lesson_examples_and_practice():
    assert set(LESSONS) == set(CATEGORIES)
    for category_id in CATEGORIES:
        html = client.get(f"/practice/{category_id}").text
        lesson = LESSONS[category_id]
        assert lesson["rule"] in html
        assert lesson["contrast"] in html
        assert len(lesson["examples"]) in (2, 3)
        for example in lesson["examples"]:
            assert example["before"] in html
            assert example["after"] in html
        assert 'id="start-practice"' in html
        assert "完成五題才會儲存紀錄" in html
        assert "尚未開放，目前不會計入" not in html
