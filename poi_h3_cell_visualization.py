import pandas as pd
from pathlib import Path

out_dir = Path("preprocessed_data/NYC")
df = pd.read_csv(out_dir / "NYC.csv")
