import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# --- PATHS YOU ALREADY USE ---
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
PANEL_FILE = "outputs/myanmar/cleaned_panel.csv"

print("Loading boundaries...")
bounds = gpd.read_file(BOUNDARIES_FILE)

print("Loading panel data...")
panel = pd.read_csv(PANEL_FILE)

# Pick one month just for visualization (first month with complete data works)
one_month = panel["period_start"].unique()[0]
print(f"Using month: {one_month}")

panel_month = panel[panel["period_start"] == one_month].copy()

# Identify the pcode column in the boundaries file
id_col = None
for c in bounds.columns:
    if "pcode" in c.lower():
        id_col = c
        break

if id_col is None:
    raise ValueError("No pcode column found in boundaries file.")

print(f"Using ID column: {id_col}")

# Reduce panel to unique township info
panel_subset = panel_month.drop_duplicates("township_id")[["township_id", "treated", "dist_epicenter_km"]]

# Merge onto boundaries
gdf = bounds.merge(panel_subset, how="left", left_on=id_col, right_on="township_id")

print("Merged boundaries + distances. Plotting...")

# --- PLOT ---
fig, ax = plt.subplots(figsize=(6, 10))
gdf.plot(column="dist_epicenter_km", cmap="viridis", legend=True, ax=ax)

# Highlight treated in red
gdf[gdf["treated"] == 1].boundary.plot(ax=ax, edgecolor="red", linewidth=1.5, label="Treated Townships")

plt.legend()
plt.title("Distance to Epicentre and Treated Townships")
plt.tight_layout()
plt.show()

print("Done.")
