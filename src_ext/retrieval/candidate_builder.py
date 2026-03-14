from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = ["Entry", "EC number", "Sequence"]


def load_sequence_table(path):
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(col).strip() for col in df.columns]

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")

    df = df[EXPECTED_COLUMNS].copy()
    df["Entry"] = df["Entry"].astype(str).str.strip()
    df["EC number"] = df["EC number"].fillna("").astype(str).str.strip()
    df["Sequence"] = df["Sequence"].fillna("").astype(str).str.strip()
    return df


def split_ec_numbers(value):
    if not value:
        return []
    ecs = [item.strip() for item in str(value).split(";")]
    return [ec for ec in ecs if ec and ec.lower() != "nan"]


def build_train_candidates(train_df):
    records = train_df.copy()
    records["ec_list"] = records["EC number"].map(split_ec_numbers)
    return records


def build_ec_catalog(train_df):
    rows = []
    for _, row in train_df.iterrows():
        for ec in split_ec_numbers(row["EC number"]):
            rows.append(
                {
                    "ec_number": ec,
                    "entry": row["Entry"],
                    "sequence": row["Sequence"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["ec_number", "support_count", "entries", "example_sequence"])

    ec_df = pd.DataFrame(rows)
    catalog = (
        ec_df.groupby("ec_number")
        .agg(
            support_count=("entry", "count"),
            entries=("entry", lambda x: list(dict.fromkeys(x))),
            example_sequence=("sequence", "first"),
        )
        .reset_index()
    )
    return catalog


def build_clean_membership_table(train_df):
    entry_to_sequence = {}
    ec_to_entries = {}

    for _, row in train_df.iterrows():
        entry = row["Entry"]
        entry_to_sequence[entry] = row["Sequence"]
        for ec in split_ec_numbers(row["EC number"]):
            if ec not in ec_to_entries:
                ec_to_entries[ec] = []
            ec_to_entries[ec].append(entry)

    rows = []
    for ec, entries in ec_to_entries.items():
        for entry in entries:
            rows.append(
                {
                    "Entry": entry,
                    "EC number": ec,
                    "Sequence": entry_to_sequence.get(entry, ""),
                }
            )

    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
