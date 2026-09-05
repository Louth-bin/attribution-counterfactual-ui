from difflib import get_close_matches

import pandas as pd


ids = (
    pd.read_csv("qualtrics/qualtrics_results_v2.0.csv", dtype=str, keep_default_na=False)[
        "participant"
    ]
    .drop_duplicates()
    .tolist()
)

queries = [
    "R_6dM4kPwwdd2TmDK",
    "R_6pJjO84Lt3faHs",
    "R_1H8nU5aMKHB",
    "R_1iEvEgz4wlo6VnO",
    "R_1ifB9gmMBnGHY",
    "R_77UVy9h8Umm3",
    "R_7QQowZQ43kZcCj7",
    "R_2z2WPEOV2HXKyY",
    "R_5faag2PFPagYi6V",
    "R_5faag2PFpaGyI6V",
]

for query in queries:
    print(f"\nquery {query}")
    for match in get_close_matches(query, ids, n=8, cutoff=0.35):
        print(" ", match)
