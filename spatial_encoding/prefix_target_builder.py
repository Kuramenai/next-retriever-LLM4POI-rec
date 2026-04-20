from __future__ import annotations

from typing import Literal
import pandas as pd


def build_prefix_target_examples(
    checkins_df: pd.DataFrame,
    config,
    *,
    mode: Literal["final_only", "sliding"] = "final_only",
    min_prefix_len: int = 1,
) -> pd.DataFrame:
    """
    Build prefix-target examples for next-POI prediction.

    Parameters
    ----------
    checkins_df:
        Raw session check-ins with at least:
        - config.session_id_col
        - config.poi_id_col
        - config.timestamp_col
        - config.category_col

    mode:
        - "final_only":
            one example per session:
            prefix = session[:-1], target = session[-1]
        - "sliding":
            one example per prediction step:
            prefix = session[:t], target = session[t]

    min_prefix_len:
        Minimum observed prefix length required to emit an example.

    Returns
    -------
    examples_df:
        One row per example with:
        - example_id
        - SessionId
        - prefix_len
        - prefix_poi_sequence
        - target_poi
        - target_category
        - target_timestamp
        - prefix_checkins   (list[dict], directly usable to rebuild a DataFrame)
    """
    required_cols = [
        config.session_id_col,
        config.poi_id_col,
        config.timestamp_col,
        config.category_col,
    ]
    missing = [c for c in required_cols if c not in checkins_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in checkins_df: {missing}")

    df = checkins_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col])
    df = df.sort_values([config.session_id_col, config.timestamp_col]).reset_index(
        drop=True
    )

    examples = []
    example_id = 0

    for session_id, session_df in df.groupby(config.session_id_col, sort=False):
        session_df = session_df.sort_values(config.timestamp_col).reset_index(drop=True)
        n = len(session_df)

        # Need at least prefix + target
        if n < (min_prefix_len + 1):
            continue

        if mode == "final_only":
            prefix_df = session_df.iloc[:-1].copy()
            target_row = session_df.iloc[-1]

            if len(prefix_df) < min_prefix_len:
                continue

            examples.append(
                {
                    "example_id": example_id,
                    config.session_id_col: session_id,
                    "prefix_len": len(prefix_df),
                    "prefix_poi_sequence": prefix_df[config.poi_id_col].tolist(),
                    "target_poi": target_row[config.poi_id_col],
                    "target_category": target_row[config.category_col],
                    "target_timestamp": target_row[config.timestamp_col],
                    "prefix_checkins": prefix_df.to_dict(orient="records"),
                }
            )
            example_id += 1

        elif mode == "sliding":
            # target index t means prefix is rows [:t], target is row [t]
            for t in range(min_prefix_len, n):
                prefix_df = session_df.iloc[:t].copy()
                target_row = session_df.iloc[t]

                examples.append(
                    {
                        "example_id": example_id,
                        config.session_id_col: session_id,
                        "prefix_len": len(prefix_df),
                        "prefix_poi_sequence": prefix_df[config.poi_id_col].tolist(),
                        "target_poi": target_row[config.poi_id_col],
                        "target_category": target_row[config.category_col],
                        "target_timestamp": target_row[config.timestamp_col],
                        "prefix_checkins": prefix_df.to_dict(orient="records"),
                    }
                )
                example_id += 1
        else:
            raise ValueError(f"Unsupported mode={mode}. Use 'final_only' or 'sliding'.")

    return pd.DataFrame(examples)


# test_examples_df = build_prefix_target_examples(
#     checkins_df=test_checkins_df,
#     config=config,
#     mode="final_only",
#     min_prefix_len=1,
# )

# train_examples_df = build_prefix_target_examples(
#     checkins_df=train_checkins_df,
#     config=config,
#     mode="sliding",
#     min_prefix_len=2,
# )

# example = train_examples_df.iloc[0]
# prefix_df = pd.DataFrame(example["prefix_checkins"])

# encoded_prefix = encode_partial_session_online(
#     partial_session_df=prefix_df,
#     poi_descriptor_df=poi_spatial_df,
#     pair_transition_df=pair_transition_df,
#     config=config,
# )
