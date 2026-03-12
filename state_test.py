import pandas as pd
df = pd.read_parquet("data/processed/fact_demand.parquet")
xx = df[df["state"] == "XX"]
non_xx = df[df["state"] != "XX"]
print(f"XX rows: {len(xx):,} ({len(xx)/len(df)*100:.1f}%)")
print(f"XX Rx: {xx['number_of_prescriptions'].sum():,.0f}")
print(f"Non-XX Rx: {non_xx['number_of_prescriptions'].sum():,.0f}")
print(f"XX share of Rx: {xx['number_of_prescriptions'].sum()/df['number_of_prescriptions'].sum()*100:.1f}%")
print(f"\nXX unique drugs: {xx['ndc_11'].nunique():,}")
print(f"Non-XX unique drugs: {non_xx['ndc_11'].nunique():,}")
print(f"\nXX mean Rx per row: {xx['number_of_prescriptions'].mean():.1f}")
print(f"Non-XX mean Rx per row: {non_xx['number_of_prescriptions'].mean():.1f}")