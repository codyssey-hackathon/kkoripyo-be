"""인스타그램 데이터 수집 + 정제.

EveryInfra 응답에는 요금 정보·프로필 사진 주소 등 분석에 필요 없는 값이 많아서,
여기서 필요한 필드만 남겨 돌려준다. (API에는 필드를 골라 받는 옵션이 없음)

- slim_* : 응답 한 건을 줄이는 순수 함수 (테스트하기 쉬움)
- fetch_* : 기능 하나 호출 + 정제
- collect_account : 한 계정의 기본 데이터를 동시에 모두 수집
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from app.integrations.everyinfra import EveryInfraClient, EveryInfraError

USERNAME_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")
HASHTAG_RE = re.compile(r"#([\w가-힣]+)")
MENTION_RE = re.compile(r"@([A-Za-z0-9._]{1,30})")

# EveryInfra 한도 (catalog 기준)
MAX_FOLLOW = 500
MAX_POSTS = 200
MAX_COMMENTS = 100
# 댓글을 동시에 몇 개까지 요청할지
COMMENT_CONCURRENCY = 8

# 진행 상황 콜백: (key, event, value)
Progress = Callable[[str, str, Any], None]


def normalize_username(value: str) -> str:
    name = value.strip().lstrip("@")
    m = re.search(r"instagram\.com/([A-Za-z0-9._]+)", name)
    name = (m.group(1) if m else name).strip("/").lower()
    if not USERNAME_RE.fullmatch(name):
        raise ValueError(f"인스타그램 아이디 형식이 아닙니다: {value}")
    return name


def _lower(v: Any) -> str | None:
    return v.lower() if isinstance(v, str) else None


# ---------- 정제 (응답 한 건 → 필요한 필드만) ----------


def slim_profile(r: dict) -> dict:
    return {
        "username": _lower(r.get("username")),
        "display_name": r.get("display_name"),
        "bio": r.get("bio"),
        "user_id": r.get("user_id"),
        "follower_count": r.get("follower_count"),
        "following_count": r.get("following_count"),
        "post_count": r.get("post_count"),
        "is_private": r.get("is_private"),
        "is_business": r.get("is_business"),
        "category": r.get("category"),
        "external_url": r.get("external_url"),
    }


def slim_user(r: dict) -> dict:
    return {
        "username": _lower(r.get("username")),
        "name": r.get("name"),
        "user_id": r.get("id"),
        "is_private": r.get("is_private"),
    }


def slim_post(r: dict) -> dict:
    text = r.get("text") or ""
    raw_id = str(r.get("id") or "")
    return {
        "post_id": r.get("post_id"),  # 주소에 쓰는 코드 (/p/<post_id>/)
        "media_id": raw_id.split("_")[0] or None,  # 인스타 내부 숫자 ID ("<media>_<user>")
        "url": r.get("url"),
        "author": _lower(r.get("author_username")),
        "posted_at": r.get("posted_at"),
        "media_type": r.get("media_type"),
        "text": text,
        "hashtags": r.get("hashtags") or HASHTAG_RE.findall(text),
        "mentions": [m.rstrip(".").lower() for m in MENTION_RE.findall(text)],
        "location": r.get("location"),
        "image_url": r.get("image_url"),  # 약 5일 뒤 만료. 여러 장 게시물은 첫 장만
        "video_url": r.get("video_url"),
        "like_count": r.get("like_count"),
        "comment_count": r.get("comment_count"),
    }


def slim_comment(r: dict) -> dict:
    text = r.get("text") or ""
    return {
        "comment_id": r.get("id"),
        "username": _lower(r.get("author_username")),
        "text": text,
        "posted_at": r.get("posted_at"),
        "like_count": r.get("like_count"),
        "mentions": [m.rstrip(".").lower() for m in MENTION_RE.findall(text)],
    }


# ---------- 호출 + 정제 ----------


async def fetch_profile(ei: EveryInfraClient, username: str) -> dict:
    r = await ei.call("profile", {"username": normalize_username(username)})
    if not r:
        raise EveryInfraError("not_found", f"계정을 찾을 수 없습니다: {username}")
    return slim_profile(r)


async def _fetch_follow(ei: EveryInfraClient, action: str, username: str, limit: int) -> dict:
    limit = max(1, min(limit, MAX_FOLLOW))
    rows = await ei.call(action, {"username": normalize_username(username), "limit": limit})
    users = [slim_user(x) for x in rows]
    # limit만큼 꽉 찼으면 실제로는 더 있을 수 있음
    return {"count": len(users), "maybe_more": len(users) >= limit, "users": users}


async def fetch_followers(ei: EveryInfraClient, username: str, limit: int = MAX_FOLLOW) -> dict:
    return await _fetch_follow(ei, "followers_list", username, limit)


async def fetch_following(ei: EveryInfraClient, username: str, limit: int = MAX_FOLLOW) -> dict:
    return await _fetch_follow(ei, "following_list", username, limit)


async def fetch_user_posts(ei: EveryInfraClient, username: str, limit: int = 30) -> list[dict]:
    limit = max(1, min(limit, MAX_POSTS))
    rows = await ei.call("user_posts", {"username": normalize_username(username), "limit": limit})
    return [slim_post(x) for x in rows]


async def fetch_post_comments(
    ei: EveryInfraClient, post_url: str, limit: int = MAX_COMMENTS
) -> list[dict]:
    if not re.match(r"^https://www\.instagram\.com/(p|reel)/[A-Za-z0-9_-]+/?$", post_url):
        raise ValueError(f"게시물 주소 형식이 아닙니다: {post_url}")
    limit = max(1, min(limit, MAX_COMMENTS))
    rows = await ei.call("comments", {"url": post_url, "limit": limit})
    return [slim_comment(x) for x in rows]


# ---------- 한 계정 통째로 ----------


async def collect_account(
    ei: EveryInfraClient,
    username: str,
    *,
    posts_limit: int = 30,
    follow_limit: int = MAX_FOLLOW,
    comments_limit: int = MAX_COMMENTS,
    with_comments: bool = True,
    progress: Progress | None = None,
) -> dict:
    """프로필·팔로워·팔로잉·게시물을 동시에 요청하고, 게시물이 오면 댓글도 동시에 요청한다.

    일부가 실패해도 나머지는 돌려주고, 실패한 부분은 errors에 적는다.
    progress(key, event, value): 진행 상황을 받는 함수 (화면 표시용, 선택)
      key: profile|followers|following|posts|comments
      event: start|done|error|progress
      value: done이면 받은 결과(comments는 총 개수), progress면 (완료 수, 전체 수)
    """
    say: Progress = progress or (lambda *_: None)
    username = normalize_username(username)
    parts = ("profile", "followers", "following", "posts")

    async def tracked(name: str, coro: Awaitable[Any]) -> Any:
        say(name, "start", None)
        try:
            value = await coro
        except Exception as e:
            say(name, "error", e)
            raise
        say(name, "done", value)
        return value

    results = await asyncio.gather(
        tracked("profile", fetch_profile(ei, username)),
        tracked("followers", fetch_followers(ei, username, follow_limit)),
        tracked("following", fetch_following(ei, username, follow_limit)),
        tracked("posts", fetch_user_posts(ei, username, posts_limit)),
        return_exceptions=True,
    )
    out: dict[str, Any] = {"username": username, "errors": []}
    for name, value in zip(parts, results, strict=True):
        if isinstance(value, Exception):
            out[name] = None
            out["errors"].append({"part": name, "error": str(value)})
        else:
            out[name] = value

    posts = out["posts"] or []
    if with_comments and posts:
        sem = asyncio.Semaphore(COMMENT_CONCURRENCY)
        targets = sum(1 for p in posts if p.get("comment_count"))
        done = 0
        if targets:
            say("comments", "start", targets)

        async def one(post: dict) -> list[dict]:
            nonlocal done
            if not post.get("comment_count"):  # 댓글 0개면 요청 안 함 (비용 절약)
                return []
            async with sem:
                rows = await fetch_post_comments(ei, post["url"], comments_limit)
            done += 1
            say("comments", "progress", (done, targets))
            return rows

        comment_results = await asyncio.gather(*(one(p) for p in posts), return_exceptions=True)
        for post, value in zip(posts, comment_results, strict=True):
            if isinstance(value, Exception):
                post["comments"] = None
                out["errors"].append({"part": f"comments:{post['post_id']}", "error": str(value)})
            else:
                post["comments"] = value
        if targets:
            say("comments", "done", sum(len(p.get("comments") or []) for p in posts))

    out["coverage"] = _coverage(out)
    return out


def _coverage(data: dict) -> dict:
    """프로필에 표시된 숫자 대비 실제로 받은 개수."""
    profile = data.get("profile") or {}
    posts = data.get("posts") or []
    return {
        "followers": [(data.get("followers") or {}).get("count"), profile.get("follower_count")],
        "following": [(data.get("following") or {}).get("count"), profile.get("following_count")],
        "posts": [len(posts), profile.get("post_count")],
        "comments": [
            sum(len(p.get("comments") or []) for p in posts),
            sum(p.get("comment_count") or 0 for p in posts),
        ],
    }
