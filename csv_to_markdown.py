import csv
import argparse
import pandas as pd

def csv_to_markdown(rows):
    if not rows:
        return ""
    
    # Generate Markdown table header
    header = rows[0]
    markdown = "| " + " | ".join(header) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
    
    # Generate Markdown table rows
    for row in rows[1:]:
        markdown += "| " + " | ".join(row) + " |\n"
    
    return markdown

def load_and_process_csv(csv_file_path):
    # Load CSV using pandas
    df = pd.read_csv(csv_file_path, sep="\t")
    
    if df.empty:
        return "", ""
    
    # Sort rows based on the first column (assumed to be scores)
    df = df.sort_values(by=df.columns[0], ascending=True)

    # Change the datatype of the first column to str
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    # Remove rows with NaN values
    df = df.dropna()
    
    # Select top 100 and bottom 100 rows
    top_100 = df.head(100)
    bottom_100 = df.tail(100)

    # Convert to Markdown
    top_100_markdown = csv_to_markdown([df.columns.tolist()] + top_100.values.tolist())
    bottom_100_markdown = csv_to_markdown([df.columns.tolist()] + bottom_100.values.tolist())
    
    return top_100_markdown, bottom_100_markdown

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert CSV to Markdown Tables')
    parser.add_argument('--csv_file', type=str, help='Path to the input CSV file')
    
    args = parser.parse_args()

    top_100_markdown, bottom_100_markdown = load_and_process_csv(args.csv_file)
    

    with open(f"{args.csv_file.split('.csv')[0]}_top_100.md", "w", encoding="utf-8") as f:
        f.write(top_100_markdown)

    with open(f"{args.csv_file.split('.csv')[0]}_bottom_100.md", "w", encoding="utf-8") as f:
        f.write(bottom_100_markdown)
    print("Markdown tables saved to top_100.md and bottom_100.md")