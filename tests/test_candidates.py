"""주변 계정 후보 추출과 조사 우선순위 점수 테스트. 가상 계정만 쓴다."""

import pytest

from app.candidates import extract_candidates, list_account_candidates, priority_score

SUBJECT = "minji_lee"

COLLECT = {
    "source_tool": "collect_account",
    "account_id": "@Minji_Lee",
    "data": {
        "profile": {"username": "minji_lee", "is_private": False},
        "followers": {
            "count": 2,
            "maybe_more": False,
            "users": [
                {"username": "jisoo_park", "is_private": False},
                {"username": "follow_only", "is_private": True},
            ],
        },
        "following": {"count": 1, "maybe_more": False, "users": [{"username": "follow_only"}]},
        "posts": [
            {
                "post_id": "P1",
                "text": "주말에 가평 다녀옴 with @jisoo_park",
                "mentions": ["jisoo_park"],
                "tagged_accounts": [{"username": "jisoo_park"}],
                "comments": [
                    {"username": "fan_a", "text": "부럽다", "posted_at": "t1"},
                    {"username": "fan_a", "text": "나도 갈래", "posted_at": "t2"},
                    {"username": "once_b", "text": "ㅎㅎ", "posted_at": "t3"},
                    {"username": "minji_lee", "text": "다음에 같이 가자", "posted_at": "t4"},
                ],
            },
            {"post_id": "P2", "text": "학생증 재발급", "mentions": [], "comments": []},
        ],
    },
}
STORE = {"scrape_0001": COLLECT}


def by_id(candidates):
    return {c["account_id"]: c for c in candidates}


def test_extracts_relations_with_subject_only():
    found = by_id(extract_candidates(SUBJECT, [COLLECT]))
    assert SUBJECT not in found
    assert set(found) == {"jisoo_park", "fan_a", "once_b", "follow_only"}

    jisoo = found["jisoo_park"]
    assert jisoo["relation_types"] == ["tagged", "mentioned", "follower"]
    assert jisoo["evidence_source_ids"] == ["minji_lee:P1:caption:0"]
    assert jisoo["occurrence_count"] == 2  # 근거 위치 1 + follower 1
    assert jisoo["is_private"] is False

    assert found["fan_a"]["evidence_source_ids"] == [
        "minji_lee:P1:comment:0",
        "minji_lee:P1:comment:1",
    ]
    follow_only = found["follow_only"]
    assert follow_only["relation_types"] == ["follower", "following"]
    assert follow_only["evidence_source_ids"] == []
    assert follow_only["is_private"] is True
    assert found["once_b"]["is_private"] is None


def test_priority_score_follows_table():
    found = by_id(extract_candidates(SUBJECT, [COLLECT]))
    assert found["jisoo_park"]["priority_score"] == 9  # tagged 5 + mentioned 4
    assert found["fan_a"]["priority_score"] == 2  # 댓글 2건 이상
    assert found["once_b"]["priority_score"] == 0  # 댓글 1건은 가산 안 함
    assert found["follow_only"]["priority_score"] == 0


def test_priority_score_is_capped():
    assert priority_score(["tagged", "mentioned", "commented"], 3) == 10
    # 댓글 가점은 commented 관계가 있을 때만
    assert priority_score(["follower"], 5) == 0


def test_candidates_sorted_by_priority():
    ids = [c["account_id"] for c in extract_candidates(SUBJECT, [COLLECT])]
    assert ids == ["jisoo_park", "fan_a", "follow_only", "once_b"]


def test_neighbor_posts_only_count_links_to_subject():
    neighbor_posts = {
        "source_tool": "get_user_posts",
        "account_id": "jisoo_park",
        "data": [
            {"post_id": "N1", "mentions": ["@minji_lee", "stranger"], "tagged_accounts": []},
            {"post_id": "N2", "mentions": ["stranger"], "tagged_accounts": ["minji_lee"]},
        ],
    }
    neighbor_comments = {
        "source_tool": "get_post_comments",
        "account_id": "jisoo_park",
        "post_id": "N1",
        "data": [
            {"username": "minji_lee", "text": "또 가자", "posted_at": "t9"},
            {"username": "stranger", "text": "누구야", "posted_at": "t10"},
        ],
    }
    found = by_id(extract_candidates(SUBJECT, [neighbor_posts, neighbor_comments]))
    assert set(found) == {"jisoo_park"}  # 주변 계정끼리의 관계는 후보가 아니다
    assert found["jisoo_park"]["relation_types"] == ["tagged", "mentioned", "commented"]
    assert found["jisoo_park"]["evidence_source_ids"] == [
        "jisoo_park:N1:caption:0",
        "jisoo_park:N1:comment:0",
        "jisoo_park:N2:caption:0",
    ]


def test_same_comment_from_two_tools_counted_once():
    again = {
        "source_tool": "get_post_comments",
        "account_id": SUBJECT,
        "post_id": "P2",
        "data": [{"username": "once_c", "text": "안녕", "posted_at": "t1"}],
    }
    found = by_id(extract_candidates(SUBJECT, [again, again]))
    assert found["once_c"]["occurrence_count"] == 1
    assert found["once_c"]["priority_score"] == 0


def test_follow_lists_of_neighbors_are_ignored():
    neighbor_followers = {
        "source_tool": "get_followers",
        "account_id": "jisoo_park",
        "data": [{"username": "stranger"}],
    }
    assert extract_candidates(SUBJECT, [neighbor_followers]) == []


def test_profile_privacy_overrides_follow_list():
    profile = {
        "source_tool": "get_profile",
        "account_id": "jisoo_park",
        "data": {"is_private": True},
    }
    found = by_id(extract_candidates(SUBJECT, [COLLECT, profile]))
    assert found["jisoo_park"]["is_private"] is True


def test_tool_pages_results():
    first = list_account_candidates(
        {"result_refs": ["scrape_0001"], "limit": 3}, subject=SUBJECT, results=STORE.get
    )
    assert first["status"] == "ok"
    assert first["total_count"] == 4
    assert len(first["candidates"]) == 3
    assert first["next_offset"] == 3

    second = list_account_candidates(
        {"result_refs": ["scrape_0001"], "limit": 3, "offset": 3},
        subject=SUBJECT,
        results=STORE.get,
    )
    assert [c["account_id"] for c in second["candidates"]] == ["once_b"]
    assert second["next_offset"] is None


@pytest.mark.parametrize(
    "args",
    [{}, {"result_refs": []}, {"result_refs": ["scrape_9999"]}, {"result_refs": "scrape_0001"}],
)
def test_tool_rejects_unknown_result_ref(args):
    assert list_account_candidates(args, subject=SUBJECT, results=STORE.get) == {
        "status": "error",
        "error_code": "INVALID_RESULT_REF",
    }


@pytest.mark.parametrize("page", [{"limit": 0}, {"limit": 51}, {"limit": True}, {"offset": -1}])
def test_tool_rejects_bad_pagination(page):
    response = list_account_candidates(
        {"result_refs": ["scrape_0001"], **page}, subject=SUBJECT, results=STORE.get
    )
    assert response["error_code"] == "INVALID_PAGINATION"
