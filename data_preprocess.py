import pandas as pd
import json
import os


def load_data(file_path):
    print(f"Loading data from {file_path}...")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if '_id' in df.columns and 'id' not in df.columns:
        df.rename(columns={'_id': 'id'}, inplace=True)

    for col in ['text', 'conspiracy', 'markers', 'annotator', 'id']:
        if col not in df.columns:
            df[col] = None

    df['markers_str'] = df['markers'].astype(str)

    print(f"Data loaded successfully. Total records: {len(df)}")
    return df


def save_deleted_data(df_to_save, output_path):
    """
    Appends deleted records to a JSONL file.
    """
    if df_to_save.empty:
        return

    df_save = df_to_save.copy()
    if 'markers_str' in df_save.columns:
        df_save = df_save.drop(columns=['markers_str'])

    with open(output_path, 'a', encoding='utf-8') as f:
        json_str = df_save.to_json(orient='records', lines=True, force_ascii=False)
        f.write(json_str)
        f.write('\n')

    print(f"Backup: Saved {len(df_save)} deleted records to '{output_path}'")


def clean_empty_text(df, deleted_output_path):
    """
    Identifies and removes rows with empty text content interactively.
    """
    text_col = 'text'
    empty_mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
    empty_count = empty_mask.sum()

    if empty_count > 0:
        print(f"\n--- Step 0: Found {empty_count} records with empty text ---")
        print(df[empty_mask][['text', 'conspiracy', 'markers', 'annotator', 'id']].head(5).to_string(index=False))

        user_input = input(f"Delete these {empty_count} empty records? (Enter 'Y' to delete): ")
        if user_input.strip().upper() == 'Y':
            save_deleted_data(df[empty_mask], deleted_output_path)
            df = df[~empty_mask]
            print(f"Deleted. Remaining records: {len(df)}")
        else:
            print("Skipped deletion.")
    return df


def process_exact_duplicates(df, deleted_output_path):
    """
    Logic 1: Statistics for data where 'text', 'markers', and 'conspiracy' are ALL the same.
    """
    print("\nStep 1: Checking for Exact Duplicates (Same Text, Markers, and Conspiracy)")

    subset_cols = ['text', 'markers_str', 'conspiracy']
    dup_mask_all = df.duplicated(subset=subset_cols, keep=False)
    dup_mask_to_delete = df.duplicated(subset=subset_cols, keep='first')

    count_to_delete = dup_mask_to_delete.sum()

    if count_to_delete > 0:
        print(f"Found {count_to_delete} redundant records to delete (Total involved: {dup_mask_all.sum()}).")
        print("Displaying top 5 duplicate groups:")

        dup_df = df[dup_mask_all].copy()
        unique_groups = dup_df[subset_cols].drop_duplicates().head(5)

        group_idx = 1
        for _, row_key in unique_groups.iterrows():
            print(f"\n[Group {group_idx}]")
            group_rows = dup_df[
                (dup_df['text'] == row_key['text']) &
                (dup_df['markers_str'] == row_key['markers_str']) &
                (dup_df['conspiracy'] == row_key['conspiracy'])
                ]
            for _, row in group_rows.iterrows():
                print(f"  Text: {str(row['text'])[:100]}...")
                print(f"  Conspiracy: {row['conspiracy']}")
                print(f"  Markers: {row['markers']}")
                print(f"  Annotator: {row['annotator']}")
                print(f"  ID: {row['id']}")
                print("-" * 30)
            group_idx += 1

        user_input = input(
            f">>> Delete these redundant copies (keep 1, remove {count_to_delete})? (Enter 'Y' to delete): ")
        if user_input.strip().upper() == 'Y':
            save_deleted_data(df[dup_mask_to_delete], deleted_output_path)
            df = df[~dup_mask_to_delete]
            print(f"Deleted. Remaining records: {len(df)}")
        else:
            print("Skipped deletion.")
    else:
        print("No exact duplicates found.")

    return df


def process_conflicting_duplicates(df, output_path):
    """
   Statistics for data where 'text' is the same, but 'conspiracy' OR 'markers' are different.
    """
    print("\nStep 2: Checking for Conflicting Duplicates")

    text_dup_mask = df.duplicated(subset=['text'], keep=False)

    if text_dup_mask.sum() == 0:
        print("No conflicting duplicates found.")
        return df

    conflicting_df = df[text_dup_mask].copy()
    total_conflict_count = len(conflicting_df)

    print(f"Found {total_conflict_count} records with conflicting information.")

    # Display top 5 groups
    print("Displaying top 5 conflicting groups:")
    unique_texts = conflicting_df['text'].unique()[:5]
    for i, txt in enumerate(unique_texts):
        print(f"\n[Conflict Group {i + 1}]")
        group = conflicting_df[conflicting_df['text'] == txt]
        for _, row in group.iterrows():
            print(f"  Text: {str(row['text'])[:100]}...")
            print(f"  Conspiracy: {row['conspiracy']}")
            print(f"  Markers: {row['markers']}")
            print(f"  Annotator: {row['annotator']}")
            print(f"  ID: {row['id']}")
            print("-" * 30)

    # Categorization
    rows_diff_conspiracy = []
    rows_diff_markers = []
    rows_diff_both = []

    grouped = conflicting_df.groupby('text')
    for text_val, group in grouped:
        unique_cons = group['conspiracy'].nunique()
        unique_mark = group['markers_str'].nunique()
        group_items = group.to_dict('records')

        if unique_cons > 1 and unique_mark > 1:
            rows_diff_both.extend(group_items)
        elif unique_cons > 1:
            rows_diff_conspiracy.extend(group_items)
        elif unique_mark > 1:
            rows_diff_markers.extend(group_items)
        else:
            pass

    df_diff_conspiracy = pd.DataFrame(rows_diff_conspiracy)
    df_diff_markers = pd.DataFrame(rows_diff_markers)
    df_diff_both = pd.DataFrame(rows_diff_both)

    print("\nConflict Categories Statistics:")
    print(f"  1. Conspiracy info is different: {len(df_diff_conspiracy)} records")
    print(f"  2. Markers info is different:    {len(df_diff_markers)} records")
    print(f"  3. Both are different:           {len(df_diff_both)} records")
    print("-" * 50)

    print("Do you want to SAVE these records and REMOVE them from the main dataset?")
    main_input = input(">>> Proceed with save & remove? (Y/N): ")

    if main_input.strip().upper() == 'Y':
        print("\nDo you want to split the saved data into 3 separate JSON files based on the categories above?")
        print("  Y = Save 3 files (e.g., _diff_conspiracy.jsonl, _diff_markers.jsonl...)")
        print("  N = Save all to 1 single file")
        split_input = input(">>> Split files? (Y/N): ")

        base_name, ext = os.path.splitext(output_path)

        if split_input.strip().upper() == 'Y':
            # Helper function to clean and save
            def clean_save(dframe, path):
                if not dframe.empty:
                    dframe.drop(columns=['markers_str'], errors='ignore').to_json(path, orient='records', lines=True,
                                                                                  force_ascii=False)
                    print(f"  -> Saved: {path}")

            clean_save(df_diff_conspiracy, f"{base_name}_diff_conspiracy{ext}")
            clean_save(df_diff_markers, f"{base_name}_diff_markers{ext}")
            clean_save(df_diff_both, f"{base_name}_diff_both{ext}")
        else:
            if 'markers_str' in conflicting_df.columns:
                conflicting_df = conflicting_df.drop(columns=['markers_str'])
            conflicting_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
            print(f"  -> Saved all conflicting records to: {output_path}")

        df = df[~text_dup_mask]
        print(f"Removed all conflicting records from main dataset. Remaining records: {len(df)}")
    else:
        print("Skipped saving/removing conflicting data.")

    return df


def save_final_dataset(df, output_path):
    """
    Saves the final preprocessed dataframe to a JSONL file.
    """
    if 'markers_str' in df.columns:
        df = df.drop(columns=['markers_str'])

    print(f"\nSaving final cleaned dataset to {output_path}...")
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print(f"Done. Final file contains {len(df)} records.")


def create_binary_classification_dataset(cleaned_file_path, diff_markers_file_path, output_path):
    """
    Merges the 'train_cleaned_final' dataset with the 'train_conflicting_data_diff_markers' dataset.
    Since 'diff_markers' data only has conflicts in marker spans but not in the conspiracy label,
    they are valid consistent samples for binary classification (Text -> Conspiracy Label).
    Deduplicates based on 'text' to ensure unique samples for training.
    """
    print(f"\n--- Creating Binary Classification Dataset ({output_path}) ---")

    dfs = []

    # 1. Load the Cleaned Final Dataset
    if os.path.exists(cleaned_file_path):
        print(f"Loading {cleaned_file_path}...")
        dfs.append(pd.read_json(cleaned_file_path, lines=True))
    else:
        print(f"Warning: {cleaned_file_path} not found.")

    # 2. Load the Diff Markers Dataset
    if os.path.exists(diff_markers_file_path):
        print(f"Loading {diff_markers_file_path}...")
        dfs.append(pd.read_json(diff_markers_file_path, lines=True))
    else:
        print(f"Warning: {diff_markers_file_path} not found. (Did you choose to split files in Step 2?)")

    if not dfs:
        print("No data available to create binary dataset.")
        return

    # 3. Merge
    merged_df = pd.concat(dfs, ignore_index=True)

    # 4. Deduplicate (Keep only unique texts for binary classification to avoid bias)
    before_count = len(merged_df)
    merged_df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    after_count = len(merged_df)

    print(f"Merged and Deduplicated (Text-based): {before_count} -> {after_count} records.")

    merged_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print(f"Done. Binary classification dataset saved to {output_path}")


if __name__ == "__main__":
    # File configurations
    train_file = "train_rehydrated.jsonl"

    # Final Outputs
    final_output_file = "train_cleaned_final.jsonl"
    conflicting_output_base = "train_conflicting_data.jsonl"
    deleted_output_file = "train_deleted_data.jsonl"
    diff_markers_file = "train_conflicting_data_diff_markers.jsonl"
    binary_training_file = "train_cleaned_final_binary.jsonl"

    # Initialize
    if os.path.exists(deleted_output_file):
        os.remove(deleted_output_file)
        print(f"Initialized empty backup file: {deleted_output_file}")

    df = load_data(train_file)

    if not df.empty:
        df = clean_empty_text(df, deleted_output_file)
        df = process_exact_duplicates(df, deleted_output_file)
        df = process_conflicting_duplicates(df, conflicting_output_base)

        save_final_dataset(df, final_output_file)

        create_binary_classification_dataset(final_output_file, diff_markers_file, binary_training_file)