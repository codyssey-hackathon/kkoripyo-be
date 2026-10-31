"""EveryInfra 호출·정제·에이전트 틀 테스트. 실제 API는 부르지 않고 가짜 응답을 쓴다."""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.agent import graph as agent_graph
from app.agent import tools as agent_tools
from app.collect import instagram as ig
from app.integrations.everyinfra import EveryInfraClient, EveryInfraError

POST = {
    "post_id": "DdoszTJGzR8",
    "id": "3992638108957619324_61954491015",
    "url": "https://www.instagram.com/p/DdoszTJGzR8/",
    "text": "오랜만이다 #카페 @Friend.One",
    "media_type": "image",
    "like_count": 77,
    "comment_count": 1,
    "author_username": "Tester",
    "posted_at": "2026-09-23T16:06:14+00:00",
    "image_url": "https://cdn.example/x.jpg",
    "width": 100,
}


def fake_api(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":  # 작업 결과 조회
        return httpx.Response(200, json={"status": "succeeded", "results": [POST]})
    body = json.loads(request.content)
    action = body["action"]
    if action == "profile":
        res = {"username": "tester", "bio": "안녕", "follower_count": 1, "avatar_url": "x"}
        return httpx.Response(200, json={"results": res, "billing": {}})
    if action == "followers_list":
        user = {"id": "1", "username": "Friend.One", "name": "F", "is_private": False}
        return httpx.Response(200, json={"results": [user], "count": 1})
    if action == "following_list":
        return httpx.Response(200, json={"results": [], "count": 0})
    if action == "user_posts":  # 비동기 작업
        return httpx.Response(202, json={"status": "running", "job_id": "job_1"})
    if action == "comments":
        c = {"id": "9", "text": "hi @tester", "author_username": "Friend.One", "like_count": 0}
        return httpx.Response(200, json={"results": [c]})
    return httpx.Response(422, json={"error": {"code": "unknown_param", "message": "bad"}})


def client() -> EveryInfraClient:
    return EveryInfraClient("test-key", poll_interval=0, transport=httpx.MockTransport(fake_api))


def test_normalize_username():
    assert ig.normalize_username("@Nasa") == "nasa"
    assert ig.normalize_username("https://www.instagram.com/nasa/") == "nasa"
    with pytest.raises(ValueError):
        ig.normalize_username("bad name!")


def test_slim_post_keeps_only_needed_fields():
    p = ig.slim_post(POST)
    assert p["media_id"] == "3992638108957619324"
    assert p["hashtags"] == ["카페"]
    assert p["mentions"] == ["friend.one"]
    assert p["author"] == "tester"
    assert "width" not in p


def test_collect_account_runs_all_parts():
    async def go():
        async with client() as ei:
            return await ig.collect_account(ei, "tester")

    data = asyncio.run(go())
    assert data["errors"] == []
    assert data["profile"]["bio"] == "안녕"
    assert "avatar_url" not in data["profile"]
    assert data["followers"]["users"][0]["username"] == "friend.one"
    assert data["posts"][0]["comments"][0]["mentions"] == ["tester"]
    assert data["coverage"]["comments"] == [1, 1]


def test_api_error_is_raised():
    async def go():
        async with client() as ei:
            await ei.call("nope", {})

    with pytest.raises(EveryInfraError) as e:
        asyncio.run(go())
    assert e.value.code == "unknown_param"


def test_tool_returns_error_instead_of_raising(monkeypatch):
    monkeypatch.setattr(agent_tools, "make_client", client)
    out = asyncio.run(agent_tools.get_profile.ainvoke({"username": "bad name!"}))
    assert "error" in json.loads(out)


class ScriptedLLM:
    """정해진 답을 순서대로 내는 가짜 LLM (OpenAI 없이 그래프 흐름만 확인)."""

    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return self.replies.pop(0)


def test_agent_graph_calls_tools_then_reports(monkeypatch):
    monkeypatch.setattr(agent_tools, "make_client", client)
    llm = ScriptedLLM(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "collect_account", "args": {"username": "tester"}, "id": "c1"},
                    {"name": "get_profile", "args": {"username": "friend.one"}, "id": "c2"},
                ],
            ),
            AIMessage("리포트: 카페를 자주 감"),
        ]
    )
    result = asyncio.run(agent_graph.run_agent("tester", llm=llm))
    assert result["report"] == "리포트: 카페를 자주 감"
    assert [s["tool"] for s in result["tool_calls"]] == ["collect_account", "get_profile"]
    assert result["tool_errors"] == 0
