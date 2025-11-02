import numpy as np, pandas as pd, json
valid_rows = np.load("valid_rows.npy", allow_pickle=True)

rows = []
for r in valid_rows:
    if isinstance(r, dict):
        rows.append(r)
    elif isinstance(r, (pd.Series,)):
        rows.append(r.to_dict())
    else:
        try:
            rows.append(dict(r))
        except:
            rows.append({})
with open("valid_rows.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)
print("✅ valid_rows.json created successfully!")
