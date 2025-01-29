import sys
from tqdm import tqdm

def remove_duplicates(file_name):
    # Open the file and read all lines
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Create a set to store unique lines
    unique_lines = set()
    
    # Use tqdm to show progress while processing lines
    print(f"Checking for duplicates in {file_name}...")
    with open(file_name, 'w') as file:
        for line in tqdm(lines, desc="Removing duplicates", unit="line"):
            # If line isn't in set, add to set and write to file
            if line not in unique_lines:
                unique_lines.add(line)
                file.write(line)

if __name__ == "__main__":
    file_name = "data.txt"
    remove_duplicates(file_name)
