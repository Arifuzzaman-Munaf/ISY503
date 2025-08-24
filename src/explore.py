import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_label_distribution(labels, split):
    """
    Plot histogram of class distribution for labels {-1,0,1}.
    
    -1 = unlabeled
     0 = negative
     1 = positive
    """
    class_map = {0: "negative", 1: "positive"}
    class_names = [class_map[i] for i in sorted(class_map.keys())]

    # Count occurrences aligned with {0,1}
    counts = pd.Series(labels).value_counts().reindex(sorted(class_map.keys()), fill_value=0)

    plt.figure(figsize=(5,5))
    bars = plt.bar(class_names, counts.values, color="skyblue", edgecolor="black")

    # Annotate counts above bars
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 str(count), ha="center", va="bottom", fontsize=8)

    plt.ylim(0, max(counts.values) * 1.4)
    plt.title(f"{split} Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Examples")
    plt.tight_layout()
    plt.show()