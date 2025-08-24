from sklearn.model_selection import train_test_split

def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split dataframe into train and test sets with stratification.
    Shuffles the data before splitting.
    """
    # Shuffle first
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    return train_df, test_df


def print_shape(df, split=""):
    print(f"{split} Dataset has {df.shape[0]} rows and {df.shape[1]} columns")
