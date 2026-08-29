import csv
import os
import shlex
import sys

def filter_csv_columns():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    output_csv_path = "out.csv"

    if not os.path.exists(input_csv_path):
        print(f"Error: File '{input_csv_path}' not found.")
        sys.exit(1)

    # 1. Read header from CSV
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        try:
            headers = next(reader)
        except StopIteration:
            print("Error: The CSV file is empty.")
            sys.exit(1)

    # Display available columns with numbers
    print("\n--- Available Columns ---")
    for idx, col in enumerate(headers, 1):
        print(f" [{idx}] {col}")
    print("-------------------------\n")

    # 2. Await user input
    user_input = input(
        "Enter space-separated column names or numbers (1-indexed) to keep\n"
        "(e.g., 1 3 \"Header Name\" 5):\n> "
    )

    try:
        tokens = shlex.split(user_input)
    except ValueError as e:
        print(f"Error parsing input: {e}")
        sys.exit(1)

    if not tokens:
        print("No columns specified. Exiting.")
        sys.exit(0)


    # 3. Resolve tokens (names or 1-based indices) to exact column names
    selected_columns = []
    invalid_tokens = []

    for token in tokens:
        if token.isdigit():
            col_num = int(token)
            if 1 <= col_num <= len(headers):
                selected_columns.append(headers[col_num - 1])
            else:
                invalid_tokens.append(f"{token} (Out of range 1-{len(headers)})")
        else:
            if token in headers:
                selected_columns.append(token)
            else:
                invalid_tokens.append(token)

    if invalid_tokens:
        print("\nError: The following input(s) could not be resolved to valid columns:")
        for inv in invalid_tokens:
            print(f" - {inv}")
        sys.exit(1)

    print("got: ", selected_columns)
    print("reading...")

    # 4. Extract selected columns and write to out.csv
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=selected_columns, extrasaction='ignore')

        writer.writeheader()
        for row in reader:
            writer.writerow(row)

    print(f"\nSuccess! Filtered CSV written to '{output_csv_path}'.")

if __name__ == "__main__":
    filter_csv_columns()
