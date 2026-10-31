"""저장된 SNS 수집 결과에서 본인과 연결된 주변 계정 후보를 뽑고 조사 우선순위 점수를 매긴다.

본인과의 관계만 추출한다. 주변 계정끼리의 언급·댓글·팔로우는 후보로 만들지 않는다
(주변 계정 자체의 관계망을 만들지 않기 위해서다).

수집 결과 하나는 어댑터가 파싱·검증해 저장한 {source_tool, account_id, data[, post_id]} 형태다.
  - collect_account: data = {profile, followers, following, posts}
  - get_profile: data = 프로필 dict
  - get_followers / get_following: data = [{username, is_private, ...}]
  - get_user_posts: data = [게시물]
  - get_post_comments: data = [{username, text, posted_at, ...}], post_id = 댓글을 가져온 게시물 ID
게시물은 {post_id, mentions, tagged_accounts, comments} 필드를 쓴다. 계정은 'name' 문자열이나 {username} 객체 모두 받는다.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.cross_inference.schema import account_key

from .table import COMMENT_MIN, COMMENT_POINTS, MAX_PRIORITY, RELATION_POINTS

# 출력 순서. 앞의 셋만 본인과의 연결 근거이고 follower·following은 탐색 후보다
RELATIONS = ("tagged", "mentioned", "commented", "follower", "following")


def priority_score(relation_types: Iterable[str], comment_count: int) -> int:
    """tagged +5, mentioned +4, 본인과 주고받은 댓글 2건 이상 +2, 최대 10점."""
    relations = set(relation_types)
    score = sum(RELATION_POINTS.get(r, 0) for r in relations)
    if "commented" in relations and comment_count >= COMMENT_MIN:
        score += COMMENT_POINTS
    return min(MAX_PRIORITY, score)


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _handle(entry) -> str:
    if isinstance(entry, dict):
        entry = entry.get("username")
    return account_key(entry) if isinstance(entry, str) else ""


class _Collector:
    def __init__(self, subject: str):
        self.subject = account_key(subject)
        self.found: dict[str, dict] = {}
        self.private: dict[str, bool] = {}
        # 같은 댓글이 collect_account와 get_post_comments로 두 번 들어와도 한 번만 센다
        self.seen_comments: set[tuple] = set()

    def add(self, account: str, kind: str, source_id: str | None = None) -> None:
        if not account or account == self.subject:
            return
        hit = self.found.setdefault(
            account, {"relations": set(), "sources": set(), "comments": set()}
        )
        hit["relations"].add(kind)
        if source_id:
            hit["sources"].add(source_id)
            if kind == "commented":
                hit["comments"].add(source_id)

    def follows(self, kind: str, users) -> None:
        for user in _list(users):
            handle = _handle(user)
            self.add(handle, kind)
            if isinstance(user, dict) and isinstance(user.get("is_private"), bool):
                self.private.setdefault(handle, user["is_private"])

    def profile(self, owner: str, profile) -> None:
        # 프로필에서 직접 확인한 값이 팔로우 목록의 값보다 우선한다
        if isinstance(profile, dict) and isinstance(profile.get("is_private"), bool):
            self.private[owner] = profile["is_private"]

    def posts(self, owner: str, posts) -> None:
        for post in _list(posts):
            if not isinstance(post, dict) or post.get("post_id") in (None, ""):
                continue
            post_id = str(post["post_id"])
            caption = f"{owner}:{post_id}:caption:0"
            for kind, field in (("mentioned", "mentions"), ("tagged", "tagged_accounts")):
                handles = {_handle(h) for h in _list(post.get(field))}
                if owner == self.subject:
                    for handle in handles:
                        self.add(handle, kind, caption)
                elif self.subject in handles:
                    self.add(owner, kind, caption)
            self.comments(owner, post_id, post.get("comments"))

    def comments(self, owner: str, post_id: str, comments) -> None:
        """본인 게시물에 남긴 댓글은 작성자를, 본인이 남긴 댓글은 게시물 주인을 후보로 본다."""
        for n, comment in enumerate(_list(comments)):
            if not isinstance(comment, dict):
                continue
            author = _handle(comment)
            key = (owner, post_id, author, comment.get("text"), comment.get("posted_at"))
            if key in self.seen_comments:
                continue
            self.seen_comments.add(key)
            source_id = f"{owner}:{post_id}:comment:{n}"
            if owner == self.subject:
                self.add(author, "commented", source_id)
            elif author == self.subject:
                self.add(owner, "commented", source_id)

    def result(self, result: dict) -> None:
        tool, data = result.get("source_tool"), result.get("data")
        owner = account_key(str(result.get("account_id") or ""))
        if tool == "collect_account" and isinstance(data, dict):
            self.profile(owner, data.get("profile"))
            if owner == self.subject:
                for kind in ("followers", "following"):
                    block = data.get(kind)
                    users = block.get("users") if isinstance(block, dict) else None
                    self.follows("follower" if kind == "followers" else "following", users)
            self.posts(owner, data.get("posts"))
        elif tool == "get_profile":
            self.profile(owner, data)
        elif tool in ("get_followers", "get_following") and owner == self.subject:
            self.follows("follower" if tool == "get_followers" else "following", data)
        elif tool == "get_user_posts":
            self.posts(owner, data)
        elif tool == "get_post_comments" and result.get("post_id") not in (None, ""):
            self.comments(owner, str(result["post_id"]), data)


def extract_candidates(subject: str, results: Iterable[dict]) -> list[dict]:
    """수집 결과들에서 후보를 뽑아 priority_score 높은 순(같으면 등장 횟수, 계정 ID 순)으로 반환한다.

    occurrence_count는 서로 다른 근거 위치 수에 follower·following 관계 수를 더한 값이다.
    """
    collector = _Collector(subject)
    for result in results:
        if isinstance(result, dict):
            collector.result(result)
    candidates = []
    for account, hit in collector.found.items():
        relations = [r for r in RELATIONS if r in hit["relations"]]
        follow_count = sum(r in ("follower", "following") for r in relations)
        candidates.append(
            {
                "account_id": account,
                "relation_types": relations,
                "evidence_source_ids": sorted(hit["sources"]),
                "occurrence_count": len(hit["sources"]) + follow_count,
                "priority_score": priority_score(relations, len(hit["comments"])),
                "is_private": collector.private.get(account),
            }
        )
    candidates.sort(key=lambda c: (-c["priority_score"], -c["occurrence_count"], c["account_id"]))
    return candidates
