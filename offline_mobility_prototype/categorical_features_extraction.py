import pandas as pd


def extract_category_sequence(session_df: pd.DataFrame) -> list[str]:
    """
    Return the ordered category sequence for one session.
    """
    if session_df.empty:
        raise ValueError("session_df cannot be empty")

    s = session_df.copy()
    s["Time"] = pd.to_datetime(s["Time"])
    s = s.sort_values(["Time", "PId"]).reset_index(drop=True)

    categories = (
        s["Category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .tolist()
    )
    return categories


def category_sequence_to_document(category_sequence: list[str]) -> str:
    """
    Convert ordered categories into a whitespace-delimited document.
    Example:
      ['coffee_shop', 'office', 'restaurant']
      -> 'coffee_shop office restaurant'
    """
    return " ".join(category_sequence)


def build_category_documents(session_checkins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one category document per session.
    Returns columns: SessionId, category_doc
    """
    df = session_checkins_df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["SessionId", "Time", "PId"]).reset_index(drop=True)

    rows = []
    for session_id, group in df.groupby("SessionId", sort=False):
        seq = extract_category_sequence(group)
        doc = category_sequence_to_document(seq)
        rows.append(
            {
                "SessionId": session_id,
                "category_sequence": seq,
                "category_doc": doc,
            }
        )

    return pd.DataFrame(rows)
