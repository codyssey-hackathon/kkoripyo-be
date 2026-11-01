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
# 서버가 미리 피드를 읽어 둘 친한 친구 수와 친구당 게시물 수
TOP_FRIENDS = 4
FRIEND_POSTS = 12

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


TAGGED_LIMIT = 12


async def fetch_tagged_posts(
    ei: EveryInfraClient, username: str, limit: int = TAGGED_LIMIT
) -> list[dict]:
    """다른 사람이 올린 사진 중 이 계정이 태그된 게시물."""
    limit = max(1, min(limit, MAX_POSTS))
    rows = await ei.call("tagged_posts", {"username": normalize_username(username), "limit": limit})
    return [slim_post(x) for x in rows]


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
    parts = ("profile", "followers", "following", "posts", "tagged")

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
        tracked("tagged", fetch_tagged_posts(ei, username)),
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
            say("comments", "rows", rows)  # 친구 피드 미리 받기용 (화면 표시 안 함)
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


def rank_friends(data: dict, top: int = TOP_FRIENDS) -> list[dict]:
    """본인 게시물의 댓글 수 + 캡션 멘션 수로 가까운 계정을 고른다. 비공개로 알려진 계정은 뺀다.

    반환: [{username, comments, mentions}] (점수 높은 순, 최대 top명)
    """
    me = data.get("username")
    posts = data.get("posts") or []
    users = ((data.get("followers") or {}).get("users") or []) + (
        (data.get("following") or {}).get("users") or []
    )
    private = {u["username"] for u in users if u.get("is_private")}
    comments: dict[str, int] = {}
    mentions: dict[str, int] = {}
    for post in posts:
        for c in post.get("comments") or []:
            if c.get("username"):
                comments[c["username"]] = comments.get(c["username"], 0) + 1
        for m in post.get("mentions") or []:
            mentions[m] = mentions.get(m, 0) + 1
    names = [u for u in set(comments) | set(mentions) if u and u != me and u not in private]
    names.sort(key=lambda u: (-(comments.get(u, 0) + mentions.get(u, 0)), u))
    return [
        {"username": u, "comments": comments.get(u, 0), "mentions": mentions.get(u, 0)}
        for u in names[:top]
    ]


async def fetch_friend(
    ei: EveryInfraClient, username: str, posts_limit: int = FRIEND_POSTS
) -> dict:
    """친구 한 명의 프로필과 최근 게시물을 동시에 가져온다. 실패해도 예외 대신 error에 담는다.

    반환: {profile, posts, error}. 비공개 계정이면 posts는 빈 목록이다.
    """
    out: dict[str, Any] = {"profile": None, "posts": [], "error": None}
    profile, posts = await asyncio.gather(
        fetch_profile(ei, username),
        fetch_user_posts(ei, username, posts_limit),
        return_exceptions=True,
    )
    if isinstance(profile, Exception):
        out["error"] = str(profile)
    else:
        out["profile"] = profile
        if profile.get("is_private"):
            return out
    if isinstance(posts, Exception):
        out["error"] = out["error"] or str(posts)
    else:
        out["posts"] = posts
    return out


class FriendPrefetch:
    """친구 후보가 보이는 대로 피드를 미리 받아 두고, 마지막에 상위 친구만 골라 쓴다.

    캡션 멘션은 게시물이 도착할 때, 댓글 작성자는 댓글이 도착할 때 후보가 된다.
    그래서 친구 피드 수집이 본인 댓글 수집과 겹쳐 진행된다.
    미리 받는 계정 수는 max_early로 제한한다 (EveryInfra 비용).
    """

    def __init__(
        self,
        ei: EveryInfraClient,
        me: str,
        *,
        top: int = TOP_FRIENDS,
        max_early: int = TOP_FRIENDS + 2,
        posts_limit: int = FRIEND_POSTS,
        progress: Progress | None = None,
    ):
        self.ei, self.me, self.top = ei, me, top
        self.max_early, self.posts_limit = max_early, posts_limit
        self.say: Progress = progress or (lambda *_: None)
        self.private: set[str] = set()
        self.tasks: dict[str, asyncio.Task] = {}
        self.started = False

    def mark_private(self, users: list[dict]) -> None:
        self.private |= {u["username"] for u in users if u.get("is_private")}

    def offer(self, names: list[str]) -> None:
        for name in names:
            if len(self.tasks) >= self.max_early:
                return
            self._start(name)

    def _start(self, name: str) -> None:
        if not name or name == self.me or name in self.private or name in self.tasks:
            return
        if not self.started:
            self.started = True
            self.say("friends", "start", None)
        self.tasks[name] = asyncio.create_task(fetch_friend(self.ei, name, self.posts_limit))

    async def finish(self, data: dict) -> list[dict]:
        """수집이 끝난 뒤 상위 친구를 골라 결과를 모은다. 고르지 않은 미리 받기는 취소한다."""
        if data.get("followers"):
            self.mark_private(data["followers"].get("users") or [])
        if data.get("following"):
            self.mark_private(data["following"].get("users") or [])
        ranked = rank_friends(data, self.top)
        for f in ranked:
            self._start(f["username"])
        chosen = {f["username"] for f in ranked}
        for name, task in self.tasks.items():
            if name not in chosen:
                task.cancel()
        results = []
        for f in ranked:
            results.append({**f, **await self.tasks[f["username"]]})
        if self.started:
            self.say("friends", "done", results)
        return results

    def cancel(self) -> None:
        for task in self.tasks.values():
            task.cancel()


async def collect_friends(
    ei: EveryInfraClient,
    friends: list[dict],
    *,
    posts_limit: int = FRIEND_POSTS,
    progress: Progress | None = None,
) -> list[dict]:
    """정해진 친구 목록의 피드를 한꺼번에 동시에 가져온다. ({username, ...}마다 profile·posts·error 추가)"""
    say: Progress = progress or (lambda *_: None)
    if not friends:
        return []
    say("friends", "start", len(friends))
    rows = await asyncio.gather(*(fetch_friend(ei, f["username"], posts_limit) for f in friends))
    results = [{**f, **r} for f, r in zip(friends, rows, strict=True)]
    say("friends", "done", results)
    return results


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
