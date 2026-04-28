"""
Category document builder for session clustering (Module 1).

Enhanced version with taxonomy-level control and transit/neutral absorption.

Usage
-----
    # Drop-in replacement — identical to original behavior:
    cat_df = build_category_documents(checkins_df)

    # With absorption + mid-level taxonomy:
    cat_df = build_category_documents(
        checkins_df,
        taxonomy_level="mid",
        absorb_transit=True,
        absorb_neutral=True,
    )
"""

from __future__ import annotations

from typing import Literal, Optional
import pandas as pd
import numpy as np

# Import taxonomy mappings
from poi_taxonomy import (
    RAW_TO_MID,
    RAW_TO_SEGMENTATION,
    MID_TO_SEGMENTATION,
    NEUTRAL_LABEL,
    TRANSIT_LABEL,
    TRANSIT_UNRESOLVED_LABEL,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Token formatting
# ═══════════════════════════════════════════════════════════════════════════════

def _format_token(s: str) -> str:
    """Lowercase, strip, collapse whitespace to underscores."""
    return s.strip().lower().replace(" ", "_").replace("&", "and")


# ═══════════════════════════════════════════════════════════════════════════════
# Core: resolve absorbable categories at any taxonomy level
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_absorbable_at_level(
    df: pd.DataFrame,
    output_col: str,
    session_id_col: str,
    timestamp_col: str,
    seg_col: str,
    *,
    absorb_transit: bool = True,
    absorb_neutral: bool = True,
    mode: str = "retrospective",
) -> pd.Series:
    """
    Resolve transit and neutral check-ins by filling from neighbors,
    operating on any taxonomy-level column.

    The seg_col is used to IDENTIFY which rows are transit/neutral.
    The output_col is where fills are applied (could be raw, mid, or seg).

    Transit: back-fill from destination (next non-transit check-in).
    Neutral: forward-fill from preceding activity.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by [session_id_col, timestamp_col].
    output_col : str
        Column to resolve (raw_token, mid_token, or seg_token).
    seg_col : str
        Column with segmentation-level labels for identifying absorbable rows.
    mode : str
        "retrospective" or "online".

    Returns
    -------
    pd.Series
        Resolved version of output_col.
    """
    resolved = df[output_col].copy()
    original = df[output_col].copy()  # preserve originals for unresolvable fallback
    seg = df[seg_col]
    groups = df[session_id_col]

    # ── Transit: back-fill from destination ──────────────────────────
    if absorb_transit:
        transit_mask = seg == TRANSIT_LABEL

        if transit_mask.any():
            # Mask transit positions as NaN, then bfill within session
            transit_resolved = resolved.where(~transit_mask)
            transit_resolved = transit_resolved.groupby(
                groups, sort=False
            ).transform(lambda s: s.bfill())

            # Handle trailing transit (no destination to bfill from)
            trailing_nan = transit_mask & transit_resolved.isna()

            if trailing_nan.any():
                if mode == "retrospective":
                    # Fall back to forward-fill from preceding activity
                    ffill_fallback = resolved.where(~transit_mask).groupby(
                        groups, sort=False
                    ).transform(lambda s: s.ffill())
                    transit_resolved.loc[trailing_nan] = ffill_fallback.loc[trailing_nan]
                else:
                    # Online: mark as unresolved
                    transit_resolved.loc[trailing_nan] = _format_token(
                        TRANSIT_UNRESOLVED_LABEL
                    )

            # All-transit sessions: bfill and ffill both fail.
            # Keep the original raw token (e.g. "subway", "train_station")
            # instead of injecting a generic "transit" label that would
            # pollute the TF-IDF vocabulary.
            still_nan = transit_mask & transit_resolved.isna()
            if still_nan.any():
                transit_resolved.loc[still_nan] = original.loc[still_nan]

            resolved.loc[transit_mask] = transit_resolved.loc[transit_mask]

    # ── Neutral: forward-fill from preceding context ─────────────────
    if absorb_neutral:
        neutral_mask = seg == NEUTRAL_LABEL

        if neutral_mask.any():
            neutral_resolved = resolved.where(~neutral_mask)
            neutral_resolved = neutral_resolved.groupby(
                groups, sort=False
            ).transform(lambda s: s.ffill().bfill())

            # All-neutral sessions: keep original tokens
            still_nan = neutral_mask & neutral_resolved.isna()
            if still_nan.any():
                neutral_resolved.loc[still_nan] = original.loc[still_nan]

            resolved.loc[neutral_mask] = neutral_resolved.loc[neutral_mask]

    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_category_sequence(
    session_df: pd.DataFrame,
    category_col: str = "Category",
) -> list[str]:
    """
    Return the ordered category sequence for one session.

    Backward compatible: uses raw Category column by default.
    """
    if session_df.empty:
        raise ValueError("session_df cannot be empty")

    s = session_df.copy()
    s["Time"] = pd.to_datetime(s["Time"])
    s = s.sort_values(["Time", "PId"]).reset_index(drop=True)

    categories = (
        s[category_col]
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
    """
    return " ".join(category_sequence)


def build_category_documents(
    session_checkins_df: pd.DataFrame,
    *,
    taxonomy_level: Literal["raw", "mid", "seg"] = "raw",
    absorb_transit: bool = False,
    absorb_neutral: bool = False,
    mode: str = "retrospective",
    session_id_col: str = "SessionId",
    timestamp_col: str = "Time",
    category_col: str = "Category",
    warn_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Build one category document per session.

    Parameters
    ----------
    session_checkins_df : pd.DataFrame
        Check-in level data. Required columns: session_id_col, timestamp_col,
        category_col, 'PId'.
    taxonomy_level : str
        "raw" (207 categories), "mid" (17 categories), or "seg" (8 categories).
    absorb_transit : bool
        If True, transit check-ins inherit the destination's category label
        via back-fill within each session.
    absorb_neutral : bool
        If True, neutral check-ins (Building, Road, etc.) inherit the
        preceding activity's label via forward-fill.
    mode : str
        "retrospective" (offline, full sessions) or "online" (prefix only).
        Only affects trailing transit handling when absorb_transit=True.

    Returns
    -------
    pd.DataFrame
        One row per session with columns:
        - SessionId
        - category_sequence       (list[str])
        - category_sequence_raw   (list[str], always the raw sequence for reference)
        - category_doc            (str, whitespace-delimited document)
        - taxonomy_level          (str, which level was used)
        - absorbed_transit        (bool)
        - absorbed_neutral        (bool)
    """
    df = session_checkins_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([session_id_col, timestamp_col, "PId"]).reset_index(drop=True)

    # ── Always keep the raw sequence for reference / debugging ──────
    df["_raw_token"] = (
        df[category_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # ── Map to target taxonomy level ────────────────────────────────
    if taxonomy_level == "raw":
        df["_output_token"] = df["_raw_token"]
    elif taxonomy_level == "mid":
        df["_output_token"] = (
            df[category_col]
            .map(RAW_TO_MID)
            .fillna("unknown")
            .apply(_format_token)
        )
    elif taxonomy_level == "seg":
        df["_output_token"] = (
            df[category_col]
            .map(RAW_TO_SEGMENTATION)
            .fillna("unknown")
            .apply(_format_token)
        )
    else:
        raise ValueError(
            f"taxonomy_level must be 'raw', 'mid', or 'seg', got '{taxonomy_level}'"
        )

    # ── Warn about unmapped categories ──────────────────────────────
    if warn_unmapped and taxonomy_level != "raw":
        mapping = RAW_TO_MID if taxonomy_level == "mid" else RAW_TO_SEGMENTATION
        unmapped_mask = ~df[category_col].isin(mapping)
        if unmapped_mask.any():
            unmapped_cats = df.loc[unmapped_mask, category_col].unique()
            print(
                f"[build_category_documents] WARNING: {len(unmapped_cats)} unmapped "
                f"categories ({int(unmapped_mask.sum())} rows):\n  "
                + "\n  ".join(sorted(unmapped_cats)[:20])
            )

    # ── Apply absorption if requested ───────────────────────────────
    if absorb_transit or absorb_neutral:
        # Need seg-level labels to identify which rows are transit/neutral
        df["_seg_label"] = df[category_col].map(RAW_TO_SEGMENTATION).fillna("unknown")

        df["_output_token"] = _resolve_absorbable_at_level(
            df,
            output_col="_output_token",
            session_id_col=session_id_col,
            timestamp_col=timestamp_col,
            seg_col="_seg_label",
            absorb_transit=absorb_transit,
            absorb_neutral=absorb_neutral,
            mode=mode,
        )

    # ── Build per-session documents ─────────────────────────────────
    rows = []
    for session_id, group in df.groupby(session_id_col, sort=False):
        group = group.sort_values([timestamp_col, "PId"]).reset_index(drop=True)

        raw_seq = group["_raw_token"].tolist()
        output_seq = group["_output_token"].tolist()
        doc = category_sequence_to_document(output_seq)

        rows.append({
            session_id_col: session_id,
            "category_sequence": output_seq,
            "category_sequence_raw": raw_seq,
            "category_doc": doc,
            "taxonomy_level": taxonomy_level,
            "absorbed_transit": absorb_transit,
            "absorbed_neutral": absorb_neutral,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Quick diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

def compare_absorption_effect(
    session_checkins_df: pd.DataFrame,
    n_examples: int = 5,
    taxonomy_level: str = "mid",
    session_id_col: str = "SessionId",
    timestamp_col: str = "Time",
) -> None:
    """
    Print side-by-side comparison of raw vs absorbed category sequences
    for a few example sessions. Useful for sanity-checking.
    """
    raw_docs = build_category_documents(
        session_checkins_df,
        taxonomy_level=taxonomy_level,
        absorb_transit=False,
        absorb_neutral=False,
        warn_unmapped=False,
    )
    absorbed_docs = build_category_documents(
        session_checkins_df,
        taxonomy_level=taxonomy_level,
        absorb_transit=True,
        absorb_neutral=True,
        mode="retrospective",
        warn_unmapped=False,
    )

    merged = raw_docs[[session_id_col, "category_sequence"]].rename(
        columns={"category_sequence": "before"}
    ).merge(
        absorbed_docs[[session_id_col, "category_sequence"]].rename(
            columns={"category_sequence": "after"}
        ),
        on=session_id_col,
    )

    # Find sessions where absorption changed something
    merged["changed"] = merged.apply(
        lambda r: r["before"] != r["after"], axis=1
    )

    changed = merged[merged["changed"]]
    if changed.empty:
        print("No sessions were affected by absorption.")
        return

    sample = changed.head(n_examples)

    print(f"\n{'═' * 80}")
    print(f"  ABSORPTION COMPARISON ({taxonomy_level} level)")
    print(f"  {len(changed)}/{len(merged)} sessions changed "
          f"({100*len(changed)/len(merged):.1f}%)")
    print(f"{'═' * 80}")

    for _, row in sample.iterrows():
        print(f"\n  Session: {row[session_id_col]}")
        print(f"  Before:  {' → '.join(row['before'])}")
        print(f"  After:   {' → '.join(row['after'])}")

        # Highlight changes
        diffs = []
        for i, (b, a) in enumerate(zip(row["before"], row["after"])):
            if b != a:
                diffs.append(f"    pos {i}: {b} → {a}")
        if diffs:
            print("  Changes:")
            for d in diffs:
                print(d)

    print(f"\n{'═' * 80}\n")