"""에이전트가 호출하는 도구 모음.

LLM은 각 도구의 이름·설명(docstring)·입력 스키마(args_schema)만 보고 어떤 도구를 부를지 고른다.
그래서 설명에 "언제 쓰는지 / 무엇을 돌려주는지 / 비용"을 분명히 적는다.

도구는 결과를 JSON 문자열로 돌려준다. 실패해도 예외를 던지지 않고 {"error": ...}를 돌려줘서
에이전트가 실패를 보고 다음 행동을 정할 수 있게 한다.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.collect import instagram as ig
from app.integrations.everyinfra import EveryInfraClient, EveryInfraError

USERNAME_FIELD = Field(
    description="인스타그램 아이디. @ 없이 (예: nasa). 프로필 주소를 넣어도 된다.",
    min_length=1,
    max_length=100,
)


def make_client() -> EveryInfraClient:
    """테스트에서 가짜 클라이언트로 바꿔 끼울 수 있게 함수로 둔다."""
    return EveryInfraClient()


async def _run(fn: Callable[[EveryInfraClient], Awaitable[Any]]) -> str:
    try:
        async with make_client() as ei:
            data = await fn(ei)
    except (EveryInfraError, ValueError) as e:
        data = {"error": str(e)}
    return json.dumps(data, ensure_ascii=False)


# ---------- 입력 스키마 ----------


class UsernameArgs(BaseModel):
    username: str = USERNAME_FIELD


class CollectArgs(BaseModel):
    username: str = USERNAME_FIELD
    posts_limit: int = Field(30, ge=1, le=ig.MAX_POSTS, description="가져올 최근 게시물 수")


class FollowArgs(BaseModel):
    username: str = USERNAME_FIELD
    limit: int = Field(ig.MAX_FOLLOW, ge=1, le=ig.MAX_FOLLOW, description="최대 인원")


class PostsArgs(BaseModel):
    username: str = USERNAME_FIELD
    limit: int = Field(12, ge=1, le=ig.MAX_POSTS, description="가져올 최근 게시물 수")


class CommentsArgs(BaseModel):
    post_url: str = Field(description="게시물 주소. 예: https://www.instagram.com/p/DdoszTJGzR8/")
    limit: int = Field(ig.MAX_COMMENTS, ge=1, le=ig.MAX_COMMENTS, description="최대 댓글 수")


# ---------- 도구 ----------


@tool(args_schema=CollectArgs)
async def collect_account(username: str, posts_limit: int = 30) -> str:
    """분석 대상 계정의 기본 데이터를 한 번에 모두 가져온다. 분석을 시작할 때 가장 먼저 호출한다.

    내부에서 프로필·팔로워·팔로잉·게시물을 동시에 요청하고, 게시물마다 댓글도 동시에 요청한다.
    반환: {profile, followers{count,maybe_more,users}, following{...},
           posts[{post_id, media_id, url, posted_at, text, hashtags, mentions, location,
                  image_url, like_count, comment_count, comments[{username, text, posted_at}]}],
           coverage{항목: [받은 수, 프로필 표시 수]}, errors[]}
    좋아요 누른 사람 목록은 포함되지 않는다. 비공개 계정은 대부분 비어 있다.
    """
    return await _run(lambda ei: ig.collect_account(ei, username, posts_limit=posts_limit))


@tool(args_schema=UsernameArgs)
async def get_profile(username: str) -> str:
    """계정 하나의 프로필(표시 이름, 소개글 bio, 팔로워·팔로잉·게시물 수, 비공개 여부)만 가져온다.
    주변 계정이 어떤 사람/기관인지 가볍게 확인할 때 쓴다. 비용이 가장 싸다."""
    return await _run(lambda ei: ig.fetch_profile(ei, username))


@tool(args_schema=FollowArgs)
async def get_followers(username: str, limit: int = ig.MAX_FOLLOW) -> str:
    """계정의 팔로워 목록 [{username, name, user_id, is_private}]을 가져온다.
    분석 대상 본인은 collect_account에 이미 들어 있으니 다른 계정에만 쓴다."""
    return await _run(lambda ei: ig.fetch_followers(ei, username, limit))


@tool(args_schema=FollowArgs)
async def get_following(username: str, limit: int = ig.MAX_FOLLOW) -> str:
    """계정이 팔로우하는 목록 [{username, name, user_id, is_private}]을 가져온다.
    분석 대상 본인은 collect_account에 이미 들어 있으니 다른 계정에만 쓴다."""
    return await _run(lambda ei: ig.fetch_following(ei, username, limit))


@tool(args_schema=PostsArgs)
async def get_user_posts(username: str, limit: int = 12) -> str:
    """다른 계정(예: 댓글을 자주 단 친구, 태그된 계정)의 최근 게시물을 가져온다.
    반환: [{post_id, url, posted_at, text, hashtags, mentions, location, image_url,
           like_count, comment_count}]. 댓글은 포함하지 않는다(필요하면 get_post_comments).
    분석 대상과 연결 단서가 있는 계정에만 쓴다. 수십 초 걸릴 수 있다."""
    return await _run(lambda ei: ig.fetch_user_posts(ei, username, limit))


@tool(args_schema=CommentsArgs)
async def get_post_comments(post_url: str, limit: int = ig.MAX_COMMENTS) -> str:
    """게시물 하나의 댓글 [{username, text, posted_at, like_count, mentions}]을 가져온다.
    get_user_posts로 본 다른 계정 게시물의 댓글을 더 보고 싶을 때 쓴다."""
    return await _run(lambda ei: ig.fetch_post_comments(ei, post_url, limit))


# 에이전트에게 주는 도구. 분석 대상 본인 수집(collect_account)은 그래프의 collect 노드에서
# 서버가 먼저 실행하므로 여기에는 넣지 않는다. (도구 정의는 명세 1절 6종 그대로 유지)
TOOLS = [
    get_profile,
    get_followers,
    get_following,
    get_user_posts,
    get_post_comments,
]
