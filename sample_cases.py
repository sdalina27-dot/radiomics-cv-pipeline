import argparse
from pathlib import Path

import pandas as pd


def find_label_column(df: pd.DataFrame, requested_label_column: str) -> str:
    if requested_label_column in df.columns:
        return requested_label_column

    lowered = {str(col).lower(): col for col in df.columns}
    if requested_label_column.lower() in lowered:
        return lowered[requested_label_column.lower()]

    raise ValueError(f"找不到 label column：{requested_label_column}")


def split_cases(
    input_csv: Path,
    output_dir: Path,
    label_column: str,
    per_label_count: int,
    seed: int | None,
) -> tuple[Path, Path]:
    df = pd.read_csv(input_csv)
    actual_label_column = find_label_column(df, label_column)

    sampled_parts = []
    for label_value in [1, 0]:
        label_df = df[df[actual_label_column] == label_value]
        if len(label_df) < per_label_count:
            raise ValueError(
                f"Label={label_value} 只有 {len(label_df)} 筆，不足抽選 {per_label_count} 筆"
            )
        sampled_parts.append(
            label_df.sample(n=per_label_count, random_state=seed).copy()
        )

    sampled_df = pd.concat(sampled_parts)
    remaining_df = df.drop(index=sampled_df.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_path = output_dir / "sampled_cases_10.csv"
    remaining_path = output_dir / "remaining_cases_70.csv"

    sampled_df.to_csv(sampled_path, index=False)
    remaining_df.to_csv(remaining_path, index=False)

    return sampled_path, remaining_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="隨機抽選 label=1 與 label=0 各 5 筆，並輸出 sampled/remaining 兩個 CSV。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Cine_output_20260606_binWidth_50.csv"),
        help="原始 CSV 路徑；不會修改此檔案",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="輸出目錄；預設為目前目錄",
    )
    parser.add_argument(
        "--label-column",
        default="Label",
        help="label column 名稱；預設 Label，也支援小寫 label",
    )
    parser.add_argument(
        "--per-label-count",
        type=int,
        default=5,
        help="每個 label 要抽幾筆；預設 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="可選 random seed；不指定則每次隨機不同",
    )
    args = parser.parse_args()

    sampled_path, remaining_path = split_cases(
        input_csv=args.input,
        output_dir=args.output_dir,
        label_column=args.label_column,
        per_label_count=args.per_label_count,
        seed=args.seed,
    )

    print(f"已建立：{sampled_path}")
    print(f"已建立：{remaining_path}")


if __name__ == "__main__":
    main()
