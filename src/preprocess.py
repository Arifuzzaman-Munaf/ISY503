import re
import html
import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')

# regex for URLs
url_regex = re.compile(r'http\S+|www\S+|https\S+') 

# Regex for HTML tags
HTML_TAG_RE = re.compile(r"</?(\w+)[^>]*>")

# punctuation set and regex
PUNCTUATION = string.punctuation
PUNCT_RE = re.compile(rf"[{re.escape(PUNCTUATION)}]")

def check_html_count(df, column_name, split_name, n_samples=5):
    """
    Count HTML tags in df[column_name] and show examples.

    Returns a small DataFrame with the first n_samples rows that contain HTML tags,
    including counts per row, for inspection.
    """
    s = df[column_name].astype(str)
    tag_counts = s.str.count(HTML_TAG_RE)
    any_html = tag_counts > 0

    print(f"[{split_name}] Total rows: {len(df)}")
    print(f"[{split_name}] Rows with HTML tags: {int((tag_counts > 0).sum())}\n")

    # Show up to n_samples examples that contain HTML
    examples = df.loc[any_html, [column_name]].copy().head(n_samples)
    examples["html_tag_count"] = tag_counts[any_html].head(n_samples).values

    print(f"Sample {min(n_samples, len(examples))} rows with HTML tags in [{split_name}]:")
    with pd.option_context("display.max_colwidth", 200):
        print(examples)


def remove_html(df, column_name, split_name, inplace=False):
    """
    Remove only HTML tags in df[column_name].
    - Strips tags like <br>, <a ...>, etc.
    - Normalizes whitespace.

    If inplace=True, returns the modified DataFrame.
    If inplace=False, returns a cleaned Series.
    """
    target = df if inplace else df.copy()

    # Ensure string dtype
    s = target[column_name].astype(str)

    # Remove tags
    s = s.str.replace(HTML_TAG_RE, "", regex=True)

    # Normalize whitespace (collapse multiple spaces/newlines)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    target[column_name] = s

    print(f"Removed HTML tags from [{split_name}] data. Cleaned column: '{column_name}'.")

    return target if inplace else target[column_name]


def count_urls(df, column_name, split_name):
    """
    This function counts the number of URLs in dataframe
    df: dataframe of the dataset
    column_name: column name to count URLs
    split_name: name of the dataset
    """
    # Count URLs in values of a column
    urls = df[column_name].astype(str).str.count(url_regex)
    print(f"Number of URLs in {split_name} data: {urls.sum()}")
    print(f"Samples texts containing URLs in {split_name} data:")
    print(df.loc[urls > 0, column_name].head() if urls.sum() > 0 else "No URLs found")


def remove_urls(df, column_name, split_name):
    """
    This function removes URLs from a column of a dataframe
    df: dataframe of the dataset
    column_name: column name to remove URLs
    split_name: name of the dataset
    """
    # Remove URLs from values of a column
    df[column_name] = df[column_name].astype(str).str.replace(url_regex, '', regex=True)
    print(df[column_name].head())
    print(f"Removing URLs from {split_name} data...")
    print("Done!\n")
    
    return df[column_name] # return the dataset without URLs


def punctuation_count(df, column_name, split_name, n_samples=5):
    """
    Detect and summarize punctuation usage in a text column of a dataset.

    args:
    df : DataFrame containing the text data
    column_name : column with text values
    split_name : label describing the dataset split (e.g., "Train", "Validation")
    n_samples : number of sample rows containing punctuation to display (default=5)
    """
    s = df[column_name].astype(str)

    # Count punctuation occurrences per row
    counts = s.str.count(PUNCT_RE)

    # Print overall statistics
    print(f"[{split_name}] Total rows: {len(df)}")
    print(f"[{split_name}] Rows containing punctuation: {int((counts > 0).sum())}")
    print(f"[{split_name}] Total punctuation characters: {int(counts.sum())}\n")

    # Display sample rows containing punctuation
    if (counts > 0).any():
        print(f"Sample rows with punctuation in [{split_name}]:")
        with pd.option_context("display.max_colwidth", 200):
            print(df.loc[counts > 0, [column_name]].head(n_samples))

    return counts


def remove_punctuation(df, column_name, split_name, inplace=False):
    """
    Remove punctuation characters from a text column of a dataset.

    args:
    df : DataFrame containing the text data
    column_name : column with text values
    split_name : label describing the dataset split (e.g., "Train", "Validation")
    inplace : if True, modifies the DataFrame in place and returns it,
              otherwise returns only the cleaned column
    """
    translator = str.maketrans('', '', PUNCTUATION)
    target = df if inplace else df.copy()

    # Strip punctuation, normalize whitespace, and remove leading/trailing spaces
    target[column_name] = (
        target[column_name]
        .astype(str)
        .str.translate(translator)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print(f"Punctuation removed from [{split_name}] dataset column '{column_name}'.")
    return target if inplace else target[column_name]


def count_numbers(df, column_name, split_name):
    """
    Sum of text string in df containing any numerical digit.

    args:
        df: dataframe of the dataset
        column_name: column name to check for digits
        split_name: name of the dataset
    """
    # Use str.contains with regex pattern to find digits
    numbers = df[column_name].astype(str).str.contains('\\d')
    print(f"Number of texts containing digits in {split_name} data: {numbers.sum()}")


def remove_numbers(df, column_name, split_name):
    """
    Remove all numerical digits from the text string.

    args:
        df: dataframe of the dataset
        column_name: column name to remove digits
        split_name: name of the dataset
    """
    # Use str.replace with regex pattern to remove digits
    df[column_name] = df[column_name].astype(str).str.replace('\\d', '', regex=True)
    print(f"Removing digits from {split_name} data...")
    print("Done!\n")
    return df[column_name]


def count_non_ascii(df, column_name, split_name):
    """
    Count non-ASCII characters in a dataframe column

    Args:
        df: DataFrame containing the text data
        column_name: Name of column to check
        split_name: Name of the dataset split

    Returns:
        list: Indices of tweets containing non-ASCII characters
    """
    # Check if the tweet contains any non-ASCII characters
    # If true, return the index of the tweet
    # Regex is used to check if the tweet contains any non-ASCII characters
    non_ascii = df[column_name].str.contains(r'[^\x00-\x7F]', regex=True)

    print(f"Tweets containing non-ASCII characters in {split_name} data: {non_ascii.sum()}")

    # Return the index of the tweets containing non-ASCII characters
    return non_ascii.loc[non_ascii.values == True].index.tolist()

def remove_non_ascii(text):
    """
    Remove non-ASCII characters from text.

    args:
        text: Text string to check for non-ASCII characters

    returns:
        str: Text string with non-ASCII characters removed
    """
    return text.encode('ascii', errors='ignore').decode('ascii')

def remove_stopwords(text):
    """
    Remove stopwords from text.

    Args:
        text: Text string to remove stopwords from

    Returns:
        str: Text string with stopwords removed
    """
    # Convert text to lowercase
    text = text.lower()

    # Split text into words
    words = text.split()

    # filter out english stopwords from stopwords library
    stop_words = set(stopwords.words('english'))

    # keep only words that are not in the stopwords library
    words = [word for word in words if word not in stop_words]

    # join words back into a single string
    return ' '.join(words)


def show_word_count(df, col, split="Datset"):
  word_length = df[col].astype(str).str.split().str.len()

  # Plot histogram
  word_length.hist(bins=50)
  plt.xlabel(f"Word length count of {split} dataset")
  plt.ylabel(f"Frequency")
  plt.show()



def remove_outliers(df, text_col="text", split="Dataset", min_q=0.02, max_q=0.99, show_n=10):
    """
    Remove outliers in review length (word count) using quantiles.
    
    Steps:
      - Compute word length from the text column
      - Determine quantile cutoffs (default 1%–99%)
      - Show outlier statistics and examples
      - Return DataFrame with outliers removed
    """
    data = df.copy()
    data["word_len"] = data[text_col].astype(str).str.split().str.len()

    # Compute quantiles
    low, high = data["word_len"].quantile([min_q, max_q])

    # Apply filtering
    df_clean = data[(data["word_len"] >= low) & (data["word_len"] <= high)].copy()

    # Logging
    print(f"[{split}] Column: '{text_col}'")
    print(f"  Quantile cutoffs: low={low:.3f}, high={high:.3f}")
    print(f"  Total rows: {len(data)} | Inliers kept: {len(df_clean)} | Outliers removed: {len(data) - len(df_clean)}\n")

    # Show sample outliers
    outliers_low  = data[data["word_len"] < low].head(show_n//2)
    outliers_high = data[data["word_len"] > high].head(show_n//2)
    if not outliers_low.empty or not outliers_high.empty:
        print("Examples of outliers:")
        display(pd.concat([outliers_low, outliers_high]))
    else:
        print("No outliers found.")

    return df_clean
