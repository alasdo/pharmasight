import pandas as pd

df = pd.read_parquet("data/processed/text_regulation_corpus.parquet")
with_drugs = df[df["num_drug_mentions"] > 0]
print(f"Total docs: {len(df):,}")
print(f"With drug mentions: {len(with_drugs):,}")
print(f"\nSample drug mentions:")
for _, row in with_drugs.head(10).iterrows():
    mentions = row["drug_mentions"][:3] if isinstance(row["drug_mentions"], list) else row["drug_mentions"]
    title = str(row["title"])[:80]
    print(f"  {mentions} <- {title}")

print(f"\nEvent distribution:")
print(df["primary_event"].value_counts().to_string())

print(f"\nSentiment stats:")
print(f"  Mean: {df['sentiment_score'].mean():.3f}")
print(f"  By event type:")
for evt in df["primary_event"].unique():
    mask = df["primary_event"] == evt
    print(f"    {evt:20s}: mean={df.loc[mask, 'sentiment_score'].mean():.3f}  n={mask.sum()}")
