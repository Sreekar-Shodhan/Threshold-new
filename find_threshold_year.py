import pandas as pd

import argparse

parser = argparse.ArgumentParser(description="Find threshold achievement years for countries.")
parser.add_argument('--threshold', type=float, default=80.0, help='Threshold value (default: 80.0)')
args = parser.parse_args()
THRESHOLD = args.threshold

INPUT_CSVS = [
    "20-24-Primary_fin.csv",
    "20-24-Lower_Secondary_fin.csv",
    "20-24-Higher_Secondary_fin.csv",
    "20-24-College_comp.csv"
]

def find_years(input_csv, threshold):
    df = pd.read_csv(input_csv)
    years = df.columns[1:]  # Exclude 'Country'
    result = []
    for idx, row in df.iterrows():
        country = row[0]
        vals = row[1:]
        try:
            year_idx = next(i for i, v in enumerate(vals) if pd.to_numeric(v, errors='coerce') >= threshold)
            year = years[year_idx]
        except StopIteration:
            year = None
        result.append({"Country": country, "Year": year})
    out_df = pd.DataFrame(result)
    # Exclude countries that did not hit the threshold
    out_df = out_df[out_df['Year'].notna()]
    return out_df

if __name__ == "__main__":
    for csv in INPUT_CSVS:
        output_csv = csv.replace('.csv', f'_threshold_years_{int(THRESHOLD)}.csv')
        df_out = find_years(csv, THRESHOLD)
        df_out.to_csv(output_csv, index=False)
        print(f"Wrote output to {output_csv}")

