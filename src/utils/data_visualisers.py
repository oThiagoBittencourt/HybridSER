import os
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from collections import Counter
from typing import List, Tuple, Dict
from src.utils.utils import load_config


def find_npy_shapes(directory_path: str) -> List[Tuple[int, ...]]:
    """
    Recursively walks a directory, finds .npy files, and returns a list of their shapes.

    Args:
        directory_path (str): The path to the directory to scan.

    Returns:
        A list containing the shape tuples of all found .npy arrays.
    """
    shapes_list = []
    if not os.path.isdir(directory_path):
        print(f"Warning: Directory not found at '{directory_path}'")
        return shapes_list

    print(f"Scanning directory: {directory_path}")

    # os.walk generates the file names in a directory tree
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".npy"):
                full_path = os.path.join(root, filename)
                try:
                    # Load the array to get its shape
                    array = np.load(full_path)
                    shapes_list.append(array.shape)
                except Exception as e:
                    print(f"Error loading file {full_path}: {e}")

    return shapes_list


def plot_shape_distribution(shape_counts: Dict[Tuple[int, ...], int]):
    """
    Creates and displays a horizontal bar chart from a dictionary of shape counts.

    Args:
        shape_counts: A dictionary with shape tuples as keys and their counts as values.
    """
    if not shape_counts:
        print("No data available to plot.")
        return

    # Convert shape tuples to strings for plotting labels
    labels = [str(shape) for shape in shape_counts.keys()]
    counts = list(shape_counts.values())

    plt.figure(figsize=(12, 8))

    # A horizontal bar chart is often better for long labels
    plt.barh(labels, counts, color='steelblue')

    plt.xlabel("Count of .npy Files")
    plt.ylabel("Array Shape")
    plt.title("Distribution of .npy Array Shapes")

    # Invert y-axis to show the most common item at the top
    plt.gca().invert_yaxis()

    plt.tight_layout()  # Adjust plot to ensure everything fits

    print("Chart generated successfully.")
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    config = load_config()

    # 1. Find all .npy shapes in the target directory
    all_shapes = find_npy_shapes(config["FEATURES_DIR"])

    if all_shapes:
        # 2. Count the frequency of each unique shape
        shape_counts = Counter(all_shapes)

        print("\n--- Analysis Summary ---")
        print(f"Total .npy files found: {len(all_shapes)}")
        print(f"Unique shapes found: {len(shape_counts)}")

        print("\nCounts by shape:")
        for shape, count in shape_counts.items():
            print(f"  - Shape {shape}: {count} occurrences")

        print("\nGenerating plot...")
        # 3. Plot the results
        plot_shape_distribution(shape_counts)
    else:
        print("\nAnalysis complete. No .npy files were found.")
