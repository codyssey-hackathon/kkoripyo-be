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


FINAL = agent_graph.FINAL_MARK


def test_agent_graph_collects_first_then_calls_tools(monkeypatch):
    monkeypatch.setattr(agent_tools, "make_client", client)
    llm = ScriptedLLM(
        [
            AIMessage(
                "친구 피드도 확인해 보겠습니다.",
                tool_calls=[
                    {"name": "get_profile", "args": {"username": "friend.one"}, "id": "c1"},
                    {"name": "get_user_posts", "args": {"username": "friend.one"}, "id": "c2"},
                ],
            ),
            AIMessage(f"{FINAL}\n카페를 자주 감"),
        ]
    )
    result = asyncio.run(agent_graph.run_agent("tester", llm=llm))
    assert result["report"].startswith(FINAL)
    # 서버가 먼저 수집(collect_account) → 에이전트 도구 2개
    assert [s["tool"] for s in result["tool_calls"]] == [
        "collect_account",
        "get_profile",
        "get_user_posts",
    ]
    assert result["tool_errors"] == 0


def test_agent_reads_digest_not_raw_json(monkeypatch):
    monkeypatch.setattr(agent_tools, "make_client", client)
    seen = []

    class Spy(ScriptedLLM):
        async def ainvoke(self, messages):
            seen.extend(messages)
            return await super().ainvoke(messages)

    asyncio.run(agent_graph.run_agent("tester", llm=Spy([AIMessage(f"{FINAL}\n끝")])))
    digest = next(m.content for m in seen if getattr(m, "name", None) == "collect_account")
    assert "# 게시물" in digest and "image_url" not in digest
    assert "@tester" in seen[0].content  # 시스템 프롬프트에 고객 아이디


def test_private_account_stops_before_agent(monkeypatch):
    def private_api(request):
        if request.method == "POST" and json.loads(request.content)["action"] == "profile":
            body = {"results": {"username": "tester", "is_private": True}}
            return httpx.Response(200, json=body)
        return fake_api(request)

    monkeypatch.setattr(
        agent_tools,
        "make_client",
        lambda: EveryInfraClient("k", poll_interval=0, transport=httpx.MockTransport(private_api)),
    )
    llm = ScriptedLLM([])  # LLM은 한 번도 불리면 안 됨
    result = asyncio.run(agent_graph.run_agent("tester", llm=llm))
    assert result["stopped"] == "private"
    assert "비공개" in result["report"]


def test_stream_shows_stages_not_raw_tool_calls(monkeypatch):
    monkeypatch.setattr(agent_tools, "make_client", client)
    llm = ScriptedLLM(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "get_profile", "args": {"username": "a"}, "id": "c1"},
                    {"name": "get_profile", "args": {"username": "b"}, "id": "c2"},
                ],
            ),
            AIMessage(f"{FINAL} 끝"),
        ]
    )

    async def go():
        return [e async for e in agent_graph.stream_agent("tester", llm=llm)]

    events = asyncio.run(go())
    types = {e["type"] for e in events}
    assert "tool_call" not in types and "tool_result" not in types
    assert events[-1]["type"] == "done"
    texts = [e["text"] for e in events if e["type"] == "stage"]
    assert "팔로워 1명 확인" in texts and "댓글 1개 확인" in texts
    # 같은 응답의 같은 도구 2번은 한 줄로 묶임
    assert "친구 계정 프로필 확인 (2개 계정)" in texts
    assert not any(
        "tester" in json.dumps(e, ensure_ascii=False) for e in events if e["type"] == "stage"
    )


def test_demo_page_served_in_local():
    from fastapi.testclient import TestClient

    from app.main import app

    res = TestClient(app).get("/api/v1/agent/demo")
    assert res.status_code == 200 and "꼬리표 노출 진단" in res.text


def test_agent_is_nudged_when_it_only_plans(monkeypatch):
    """계획만 쓰고 도구를 안 부르면 끝내지 않고 다시 요청한다."""
    monkeypatch.setattr(agent_tools, "make_client", client)
    llm = ScriptedLLM(
        [
            AIMessage("친구 피드도 확인해 보겠습니다."),  # 도구 호출 없음 → nudge
            AIMessage(
                "", tool_calls=[{"name": "get_profile", "args": {"username": "x"}, "id": "c1"}]
            ),
            AIMessage(f"{FINAL}\n끝"),
        ]
    )
    result = asyncio.run(agent_graph.run_agent("tester", llm=llm))
    assert result["report"].startswith(FINAL)
    assert [s["tool"] for s in result["tool_calls"]] == ["collect_account", "get_profile"]


def test_nudge_gives_up_after_limit(monkeypatch):
    monkeypatch.setattr(agent_tools, "make_client", client)
    replies = [AIMessage("확인해 보겠습니다.") for _ in range(agent_graph.MAX_NUDGES + 1)]
    result = asyncio.run(agent_graph.run_agent("tester", llm=ScriptedLLM(replies)))
    assert result["report"] == "확인해 보겠습니다."


def test_collect_account_reports_progress():
    calls = []

    async def go():
        async with client() as ei:
            return await ig.collect_account(ei, "tester", progress=lambda *a: calls.append(a))

    asyncio.run(go())
    assert ("followers", "start", None) in calls
    assert ("comments", "progress", (1, 1)) in calls
    assert ("comments", "done", 1) in calls


def test_find_clues_spots_birthday_and_school():
    from app.agent.digest import find_clues

    data = {
        "profile": {"bio": "꿈있는 일기장"},
        "posts": [
            {
                "post_id": "P1",
                "posted_at": "2023-02-05T10:00:00+00:00",
                "text": "행복한 생일",
                "comments": [{"username": "f", "text": "생일축하"}],
            },
            {
                "post_id": "P2",
                "posted_at": "2023-04-16T10:00:00+00:00",
                "text": "대학 탐방기 #단국대",
            },
        ],
    }
    clues = find_clues(data)
    assert any("[생일] 2023-02-05" in c and "P1 캡션" in c for c in clues)
    assert any("[학교·학과]" in c and "P2" in c for c in clues)


def test_account_clues_from_follow_lists():
    from app.agent.digest import build_digest, find_account_clues

    def users(*pairs):
        return {"count": len(pairs), "users": [{"username": u, "name": n} for u, n in pairs]}

    data = {
        "username": "me",
        "profile": {"display_name": "남궁민"},
        "followers": users(("dku_cs", "단국대 소프트웨어학과 학생회"), ("mom_1", "엄마")),
        "following": users(("namgung_k", "남궁현"), ("kim_a", "김철수")),
        "posts": [],
    }
    clues = "\n".join(find_account_clues(data))
    assert "[학교·학과]" in clues and "@dku_cs" in clues
    assert "[가족] 팔로워·팔로잉 계정 @mom_1" in clues
    digest = build_digest(data)
    assert "@dku_cs · 단국대 소프트웨어학과 학생회 · 팔로워" in digest


def test_common_surname_is_not_family_clue():
    from app.agent.digest import find_account_clues

    data = {
        "profile": {"display_name": "김민지"},
        "followers": {"users": [{"username": "a", "name": "김기현"}]},
        "following": {"users": []},
    }
    assert not any("가족" in c for c in find_account_clues(data))


def fake_ocr_client():
    from app.integrations import ocr

    def handler(request):
        images = json.loads(request.content)["images"]
        out = [
            {**i, "isTextPresent": True, "data": "HAPPY BIRTHDAY 지수", "error": None}
            for i in images
        ]
        return httpx.Response(200, json={"images": out})

    return ocr.OcrClient("http://ocr.test", transport=httpx.MockTransport(handler))


def test_ocr_text_reaches_digest_and_clues(monkeypatch):
    from app.integrations import ocr

    monkeypatch.setattr(agent_tools, "make_client", client)
    monkeypatch.setattr(ocr, "make_ocr_client", fake_ocr_client)
    seen = []

    class Spy(ScriptedLLM):
        async def ainvoke(self, messages):
            seen.extend(messages)
            return await super().ainvoke(messages)

    async def go():
        llm = Spy([AIMessage(f"{FINAL}\n끝")])
        return [e async for e in agent_graph.stream_agent("tester", llm=llm)]

    events = asyncio.run(go())
    digest = next(m.content for m in seen if getattr(m, "name", None) == "collect_account")
    assert "사진 속 글자: HAPPY BIRTHDAY 지수" in digest
    assert "[생일]" in digest and "사진 속 글자" in digest.split("# 규칙으로")[1]
    stages = [e["text"] for e in events if e["type"] == "stage"]
    assert "사진 1장 중 1장에서 글자 발견" in stages


def test_ocr_server_down_does_not_stop_analysis(monkeypatch):
    from app.integrations import ocr

    def down(request):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(agent_tools, "make_client", client)
    monkeypatch.setattr(
        ocr,
        "make_ocr_client",
        lambda: ocr.OcrClient("http://ocr.test", transport=httpx.MockTransport(down)),
    )

    async def go():
        llm = ScriptedLLM([AIMessage(f"{FINAL}\n끝")])
        return [e async for e in agent_graph.stream_agent("tester", llm=llm)]

    events = asyncio.run(go())
    assert events[-1]["type"] == "done"
    assert any(e.get("id") == "collect:ocr" and e["state"] == "error" for e in events)


def test_demo_page_hidden_in_production_unless_enabled(monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app

    settings = get_settings()
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "demo_enabled", False)
    assert TestClient(app).get("/api/v1/agent/demo").status_code == 404

    monkeypatch.setattr(settings, "demo_enabled", True)
    res = TestClient(app).get("/api/v1/agent/demo")
    assert res.status_code == 200
    assert 'type="password"' not in res.text and "로컬 확인용" not in res.text
    assert '<input id="key" type="hidden">' in res.text


def _blur_llm(post_ids):
    calls = [
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "select_photos_to_blur",
                    "args": {
                        "photos": [
                            {"post_id": p, "reason": "수료증에 실명이 보입니다"} for p in post_ids
                        ]
                    },
                    "id": "b1",
                }
            ],
        )
    ]
    return ScriptedLLM([*calls, AIMessage(f"{FINAL}\n끝")])


def _fake_blur(handler):
    from app.integrations import blur

    return lambda: blur.BlurClient("http://blur.test", transport=httpx.MockTransport(handler))


def test_selected_photo_is_blurred_and_streamed(monkeypatch):
    from app.integrations import blur

    def ok(request):
        images = json.loads(request.content)["images"]
        out = [{**i, "blurred_image": "data:image/jpeg;base64,QUJD", "error": None} for i in images]
        return httpx.Response(200, json={"images": out})

    monkeypatch.setattr(agent_tools, "make_client", client)
    monkeypatch.setattr(blur, "make_blur_client", _fake_blur(ok))

    async def go():
        llm = _blur_llm(["DdoszTJGzR8", "NOT_A_POST"])
        return [e async for e in agent_graph.stream_agent("tester", llm=llm)]

    events = asyncio.run(go())
    blurred = [e for e in events if e["type"] == "blur"]
    assert len(blurred) == 1  # 없는 게시물은 무시
    b = blurred[0]
    assert b["image_id"] == "DdoszTJGzR8_0" and b["before"] == "https://cdn.example/x.jpg"
    assert b["after"].startswith("data:image/jpeg") and b["reason"] == "수료증에 실명이 보입니다"
    stages = [e for e in events if e["type"] == "stage"]
    assert any(e["id"] == "blur" and e["state"] == "done" for e in stages)
    assert not any(
        "select_photos_to_blur" in e["id"] for e in stages
    )  # 내부 단계는 진행 줄에 안 보임
    assert events[-1]["type"] == "done"
    # 블러 이벤트는 리포트 뒤에 온다
    report_at = max(i for i, e in enumerate(events) if e["type"] == "token")
    assert events.index(b) > report_at


def test_no_selection_means_no_blur(monkeypatch):
    monkeypatch.setattr(agent_tools, "make_client", client)
    result = asyncio.run(
        agent_graph.run_agent("tester", llm=ScriptedLLM([AIMessage(f"{FINAL}\n끝")]))
    )
    assert result["blurred"] == [] and result["report"].startswith(FINAL)


def test_blur_server_down_still_finishes(monkeypatch):
    from app.integrations import blur

    def down(request):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(agent_tools, "make_client", client)
    monkeypatch.setattr(blur, "make_blur_client", _fake_blur(down))
    result = asyncio.run(agent_graph.run_agent("tester", llm=_blur_llm(["DdoszTJGzR8"])))
    assert result["report"].startswith(FINAL)
    assert (
        result["blurred"][0]["after"] is None
        and "blur_unavailable" in result["blurred"][0]["error"]
    )


def test_rank_friends_by_comments_and_mentions_skips_private_and_self():
    data = {
        "username": "me",
        "followers": {"users": [{"username": "locked", "is_private": True}]},
        "following": {"users": []},
        "posts": [
            {"mentions": ["b"], "comments": [{"username": "a"}, {"username": "me"}]},
            {"mentions": [], "comments": [{"username": "a"}, {"username": "locked"}]},
            {"mentions": ["c"], "comments": [{"username": "b"}, {"username": "locked"}]},
        ],
    }
    ranked = ig.rank_friends(data, top=2)
    assert [f["username"] for f in ranked] == ["a", "b"]
    assert ranked[1] == {"username": "b", "comments": 1, "mentions": 1}


def test_friend_feeds_are_prefetched_into_digest(monkeypatch):
    asked = []

    def api(request):
        if request.method == "POST":
            body = json.loads(request.content)
            asked.append((body["action"], body["params"].get("username")))
            if body["action"] == "user_posts" and body["params"]["username"] == "friend.one":
                post = {
                    **POST,
                    "post_id": "FRIEND1",
                    "text": "생일 축하해 @tester",
                    "author_username": "friend.one",
                }
                return httpx.Response(200, json={"results": [post]})
        return fake_api(request)

    monkeypatch.setattr(
        agent_tools,
        "make_client",
        lambda: EveryInfraClient("k", poll_interval=0, transport=httpx.MockTransport(api)),
    )
    seen = []

    class Spy(ScriptedLLM):
        async def ainvoke(self, messages):
            seen.extend(messages)
            return await super().ainvoke(messages)

    async def go():
        llm = Spy([AIMessage(f"{FINAL}\n끝")])
        return [e async for e in agent_graph.stream_agent("tester", llm=llm)]

    events = asyncio.run(go())
    assert ("user_posts", "friend.one") in asked and ("profile", "friend.one") in asked
    digest = next(m.content for m in seen if getattr(m, "name", None) == "collect_account")
    assert "# 가까운 친구 피드" in digest
    assert "[FRIEND1]" in digest and "[당신 멘션]" in digest
    texts = [e["text"] for e in events if e["type"] == "stage"]
    assert "가까운 친구 1명의 피드 확인" in texts


def test_private_friend_feed_is_not_read():
    def api(request):
        if request.method == "POST":
            body = json.loads(request.content)
            if body["action"] == "profile" and body["params"]["username"] == "friend.one":
                res = {"username": "friend.one", "is_private": True}
                return httpx.Response(200, json={"results": res})
        return fake_api(request)

    async def go():
        transport = httpx.MockTransport(api)
        async with EveryInfraClient("k", poll_interval=0, transport=transport) as ei:
            return await ig.collect_friends(
                ei, [{"username": "friend.one", "comments": 2, "mentions": 0}]
            )

    [f] = asyncio.run(go())
    assert f["profile"]["is_private"] and f["posts"] == []
    from app.agent.digest import _friend_lines

    assert any(
        "비공개라 볼 수 없음" in line
        for line in _friend_lines({"username": "tester", "friends": [f]})
    )


def test_friend_feed_starts_before_comments_finish(monkeypatch):
    order = []
    real_comments = ig.fetch_post_comments
    real_friend = ig.fetch_friend

    async def slow_comments(ei, url, limit=ig.MAX_COMMENTS):
        rows = await real_comments(ei, url, limit)
        await asyncio.sleep(0.05)  # 댓글이 느리게 끝나는 상황
        order.append("comments_done")
        return rows

    async def friend(ei, name, posts_limit=ig.FRIEND_POSTS):
        order.append(f"friend_start:{name}")
        return await real_friend(ei, name, posts_limit)

    monkeypatch.setattr(ig, "fetch_post_comments", slow_comments)
    monkeypatch.setattr(ig, "fetch_friend", friend)
    monkeypatch.setattr(agent_tools, "make_client", client)
    result = asyncio.run(
        agent_graph.run_agent("tester", llm=ScriptedLLM([AIMessage(f"{FINAL}\n끝")]))
    )
    assert result["report"].startswith(FINAL)
    # 캡션 멘션(@friend.one)으로 게시물 도착 직후 친구 피드를 받기 시작함
    assert order.index("friend_start:friend.one") < order.index("comments_done")


def test_prefetch_cancels_friends_not_in_top():
    async def go():
        async with client() as ei:
            pf = ig.FriendPrefetch(ei, "tester", top=1, max_early=3)
            pf.offer(["friend.one", "other", "tester"])
            assert set(pf.tasks) == {"friend.one", "other"}  # 본인 제외
            data = {
                "username": "tester",
                "posts": [{"mentions": ["friend.one"], "comments": []}],
            }
            friends = await pf.finish(data)
            return friends, pf.tasks["other"]

    friends, other = asyncio.run(go())
    assert [f["username"] for f in friends] == ["friend.one"]
    assert other.cancelled()


def test_replayed_job_results_are_fetched():
    """같은 요청을 다시 보내면 결과 없이 succeeded + replayed만 온다. 작업 조회로 결과를 받아야 한다."""

    def api(request):
        if request.method == "GET":
            return httpx.Response(200, json={"status": "succeeded", "results": [POST]})
        body = {"status": "succeeded", "job_id": "job_1", "replayed": True}
        return httpx.Response(200, json=body)

    async def go():
        transport = httpx.MockTransport(api)
        async with EveryInfraClient("k", poll_interval=0, transport=transport) as ei:
            return await ig.fetch_user_posts(ei, "tester")

    assert [p["post_id"] for p in asyncio.run(go())] == ["DdoszTJGzR8"]
