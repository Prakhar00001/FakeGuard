# Run this inside python or create a script: backend/data/download_data.py
import pandas as pd
import numpy as np

# Generate/Format 40,000 sample records if offline
n = 40000
fake_texts = ["AMAZING PRODUCT MUST BUY BEST EVER BUY NOW!!!", "Great quality 100% recommended super fast delivery superb!!!"] * (n // 4)
real_texts = ["The item arrived in good condition. Fits well and works fine.", "Decent build quality for the price, standard delivery."] * (n // 4)

df = pd.DataFrame({
    "text_": fake_texts + real_texts,
    "label": [1] * (n // 2) + [0] * (n // 2)
})
df.to_csv("backend/data/fake_reviews_40k.csv", index=False)
print("Dataset ready: 40,000 rows.")