"""내 계정~게시물~근거 간 관계 그래프. 외부 그래프 DB 없이 BFS 구현."""

from collections import defaultdict, deque


class EvidenceGraph:
    def __init__(self, json2: dict):
        self.adj = defaultdict(list)
        self.data = json2
        for post in json2["posts"]:
            p = "post:" + post["post_id"]
            owner = "account:" + post["account_id"]
            self.add(owner, p, "authored")
            for account_id in post.get("tagged_account_ids", []):
                self.add(p, "account:" + account_id, "tagged")
        for info in json2["extracted_info"]:
            origin = (
                ("post:" + info["source_post_id"])
                if info.get("source_post_id")
                else ("account:" + info["source_account_id"])
            )
            self.add(origin, "info:" + info["info_id"], "contains")
            if info.get("subject_account_id") is not None:
                self.add(
                    "info:" + info["info_id"], "account:" + info["subject_account_id"], "about"
                )
        for rel in json2["relations"]:
            source = "account:" + rel["source_account_id"]
            target = "account:" + rel["target_account_id"]
            self.add(source, target, rel["type"])

    def add(self, a: str, b: str, kind: str) -> None:
        self.adj[a].append((b, kind))
        self.adj[b].append((a, kind))

    def path(self, start: str, end: str, max_depth: int = 5):
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            if len(path) >= max_depth + 1:
                continue
            for neighbor, _ in self.adj[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None
