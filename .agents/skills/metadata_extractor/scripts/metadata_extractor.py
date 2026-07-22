import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Usage: python metadata_extractor.py <path_to_npy_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        array = np.load(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"File: {filepath}")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    print(f"Min value: {array.min()}")
    print(f"Max value: {array.max()}")
    print(f"Mean value: {array.mean():.4f}")
    print(f"Non-zero voxels: {np.count_nonzero(array)}")

if __name__ == "__main__":
    main()
