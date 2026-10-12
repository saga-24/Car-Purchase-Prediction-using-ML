# Dataset

The project uses **`Social_Network_Ads.csv`**, a small public teaching dataset commonly used in machine-learning courses.
The file is included in this repository as `data/Social_Network_Ads.csv`.

| Property | Value |
|---|---|
| Rows | 400 (one per customer) |
| Features | `Age` (years, 18-60), `EstimatedSalary` (USD, 15,000-150,000) |
| Target | `Purchased` (0 = did not buy, 1 = bought the SUV) |
| Missing values | none |
| Class balance | 143 buyers (35.75%), 257 non-buyers |

**Check you have the right file** (these values should match):
```python
import pandas as pd
df = pd.read_csv("data/Social_Network_Ads.csv")
print(df.shape)                              # (400, 3)
print(df["Age"].mean(), df["EstimatedSalary"].mean(), df["Purchased"].mean())   # 37.655  69742.5  0.3575
```
If your copy of the file has extra columns (such as `User ID` or `Gender`), keep only `Age`, `EstimatedSalary` and `Purchased`. The CSV is included so the notebook and scripts run out of the box.