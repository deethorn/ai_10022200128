#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

import re
import pandas as pd
from src.config import CSV_FILE


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_election_dataframe():
    df = pd.read_csv(CSV_FILE)

    df.columns = [str(col).strip().replace("\xa0", " ") for col in df.columns]

    for col in ["Candidate", "Party", "Year"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["Votes_numeric"] = (
        df["Votes"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)")[0]
    )
    df["Votes_numeric"] = pd.to_numeric(df["Votes_numeric"], errors="coerce").fillna(0).astype(int)

    df["Year_clean"] = df["Year"].astype(str).str.extract(r"(\d{4})")[0]

    return df


def get_2020_winner(df: pd.DataFrame):
    year_df = df[df["Year_clean"] == "2020"].copy()

    grouped = (
        year_df.groupby(["Candidate", "Party"], as_index=False)["Votes_numeric"]
        .sum()
        .sort_values("Votes_numeric", ascending=False)
    )

    if grouped.empty:
        return None

    return grouped.iloc[0]


def answer_winner_question(query_lower: str, df: pd.DataFrame):
    winner_words = ["who won", "winner", "really won"]
    ghana_2020 = "ghana" in query_lower and "2020" in query_lower
    presidential_hint = "presidential" in query_lower or "election" in query_lower

    if any(word in query_lower for word in winner_words) and ghana_2020 and presidential_hint:
        winner = get_2020_winner(df)
        if winner is not None:
            return {
                "answer": f"{winner['Candidate']} of the {winner['Party']}",
                "source": "structured_csv"
            }

    return None


def answer_votes_question(query_lower: str, df: pd.DataFrame):
    vote_words = ["how many votes", "total votes", "number of votes", "votes obtained"]

    ghana_2020 = "ghana" in query_lower and "2020" in query_lower
    election_hint = "presidential" in query_lower or "election" in query_lower or "winner" in query_lower

    if any(word in query_lower for word in vote_words) and ghana_2020 and election_hint:
        winner = get_2020_winner(df)
        if winner is not None:
            votes = int(winner["Votes_numeric"])
            return {
                "answer": f"{winner['Candidate']} obtained {votes:,} votes.",
                "source": "structured_csv"
            }

    if "who won and how many votes" in query_lower and ghana_2020:
        winner = get_2020_winner(df)
        if winner is not None:
            votes = int(winner["Votes_numeric"])
            return {
                "answer": f"{winner['Candidate']} of the {winner['Party']} won with {votes:,} votes.",
                "source": "structured_csv"
            }

    return None


def answer_party_question(query_lower: str, df: pd.DataFrame):
    if "party" not in query_lower:
        return None

    if "winner" in query_lower and "2020" in query_lower and "ghana" in query_lower:
        winner = get_2020_winner(df)
        if winner is not None:
            return {
                "answer": winner["Party"],
                "source": "structured_csv"
            }

    candidate_name = None

    if "nana akufo addo" in query_lower or "akufo addo" in query_lower:
        candidate_name = "Nana Akufo Addo"
    elif "john mahama" in query_lower or "john dramani mahama" in query_lower or "mahama" in query_lower:
        candidate_name = "John Dramani Mahama"

    if candidate_name is None:
        return None

    person_df = df[df["Candidate"].str.contains(candidate_name, case=False, na=False)]

    if not person_df.empty:
        party = person_df["Party"].mode().iloc[0]
        return {
            "answer": party,
            "source": "structured_csv"
        }

    return None


def answer_structured_query(query: str):
    query_lower = normalize_text(query)
    df = load_election_dataframe()

    winner_answer = answer_winner_question(query_lower, df)
    if winner_answer:
        return winner_answer

    votes_answer = answer_votes_question(query_lower, df)
    if votes_answer:
        return votes_answer

    party_answer = answer_party_question(query_lower, df)
    if party_answer:
        return party_answer

    return None
