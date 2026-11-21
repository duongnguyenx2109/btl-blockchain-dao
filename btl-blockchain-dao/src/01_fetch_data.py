import os
import time
import requests
import pandas as pd
import numpy as np

SNAPSHOT_GRAPHQL_URL = "https://hub.snapshot.org/graphql"
DAO_SPACE = "ens.eth"  # Dữ liệu thật của ENS DAO (có cơ chế ủy quyền lỏng)

def run_graphql_query(query: str, variables: dict) -> dict:
    resp = requests.post(
        SNAPSHOT_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]

def fetch_proposals(space: str, limit: int = 25) -> pd.DataFrame:
    query = """
    query Proposals($space: String!, $first: Int!) {
      proposals(
        first: $first,
        skip: 0,
        where: { space_in: [$space], state: "closed" },
        orderBy: "created",
        orderDirection: desc
      ) {
        id
        title
        created
        votes
        scores_total
      }
    }
    """
    data = run_graphql_query(query, {"space": space, "first": limit})
    df = pd.DataFrame(data["proposals"])
    df["created_date"] = pd.to_datetime(df["created"], unit="s")
    return df

def fetch_votes_for_proposals(proposal_ids: list, max_votes_per_batch: int = 1000) -> pd.DataFrame:
    query = """
    query Votes($proposal_ids: [String!]!, $first: Int!, $skip: Int!) {
      votes(
        first: $first,
        skip: $skip,
        where: { proposal_in: $proposal_ids },
        orderBy: "vp",
        orderDirection: desc
      ) {
        id
        voter
        vp
        choice
        created
        proposal {
          id
        }
      }
    }
    """
    all_votes = []
    skip = 0
    while skip < 5000:
        data = run_graphql_query(query, {
            "proposal_ids": proposal_ids,
            "first": max_votes_per_batch,
            "skip": skip
        })
        batch = data["votes"]
        if not batch:
            break
        for v in batch:
            all_votes.append({
                "vote_id": v["id"],
                "proposal_id": v["proposal"]["id"],
                "voter": v["voter"].lower(),
                "vp": float(v["vp"]),
                "created": v["created"]
            })
        if len(batch) < max_votes_per_batch:
            break
        skip += max_votes_per_batch
        time.sleep(0.3)
    return pd.DataFrame(all_votes)

def gini_coefficient(values: np.ndarray) -> float:
    arr = np.sort(values[values > 0])
    if len(arr) == 0:
        return 0.0
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * arr) / (n * np.sum(arr)))

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print(f"1. Dang tai 25 de xuat gan nhat cua DAO: {DAO_SPACE}...")
    df_proposals = fetch_proposals(DAO_SPACE, limit=25)
    df_proposals.to_csv("data/real_dao_proposals.csv", index=False, encoding="utf-8-sig")

    proposal_ids = df_proposals["id"].tolist()
    print("2. Dang tai du lieu phieu bau va quyen bieu quyet (vp)...")
    df_votes = fetch_votes_for_proposals(proposal_ids)
    df_votes.to_csv("data/real_dao_votes.csv", index=False, encoding="utf-8-sig")

    df_voters = (
        df_votes.groupby("voter", as_index=False)
        .agg(
            max_vp=("vp", "max"),
            avg_vp=("vp", "mean"),
            proposals_voted=("proposal_id", "nunique")
        )
        .sort_values("max_vp", ascending=False)
    )
    df_voters["participation_rate"] = df_voters["proposals_voted"] / len(df_proposals)
    df_voters.to_csv("data/real_dao_voters.csv", index=False, encoding="utf-8-sig")

    total_vp = df_voters["max_vp"].sum()
    top5_share = df_voters.head(5)["max_vp"].sum() / total_vp
    top10_share = df_voters.head(10)["max_vp"].sum() / total_vp
    gini_vp = gini_coefficient(df_voters["max_vp"].values)
    avg_participation = df_voters["participation_rate"].mean()

    print("\n=== KET QUA THONG KE DU LIEU THAT (ENS DAO) ===")
    print(f"- Tong so de xuat phan tich: {len(df_proposals)}")
    print(f"- Tong so vi tham gia bo phieu duy nhat: {len(df_voters)}")
    print(f"- Ty le tham gia trung binh moi cu tri: {avg_participation:.2%}")
    print(f"- He so bat binh dang quyen luc (Gini VP): {gini_vp:.4f}")
    print(f"- Ty trong quyen luc Top 5 dai bieu lon nhat: {top5_share:.2%}")
    print(f"- Ty trong quyen luc Top 10 dai bieu lon nhat: {top10_share:.2%}")
    print("==> Thanh cong! Da luu 3 file CSV vao thu muc data/")