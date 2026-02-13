import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import gc

# ================= CONFIGURATION =================
MAPS_DIR = Path("data/pse/external/undp_maps")
OUTPUT_CSV = Path("data/pse/external/undp_sites_digitized.csv")
# =================================================

class MapDigitizer:
    def __init__(self, image_path, existing_data=None):
        self.image_path = image_path
        self.existing_data = existing_data if existing_data is not None else []
        self.clicks = [] # Stores (x, y, type)
        self.img = plt.imread(image_path)
        self.fig = None
        self.ax = None

    def run(self):
        print(f"\n" + "="*60)
        print(f"PROCESSING: {self.image_path.name}")
        print("="*60)
        print("INSTRUCTIONS:")
        print("1. Map will open.")
        print("2. FIRST click your 2 Reference Points (e.g. top-left, bottom-right).")
        print("3. THEN click all the Project Sites.")
        print("4. Close the window when done.")
        
        # --- PHASE 1: COLLECT ALL CLICKS (GUI) ---
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.imshow(self.img)
        self.update_title()
        
        # Connect the click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        # Maximize window
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'full_screen_toggle'):
                manager.full_screen_toggle()
            elif hasattr(manager, 'window'):
                # For MacOSX backend specifically
                try:
                    manager.window.showMaximized()
                except:
                    pass
        except:
            pass
            
        plt.show() # This blocks until the window is closed
        
        # --- PHASE 2: PROCESSING (Terminal) ---
        # Force cleanup immediately after window closes
        plt.close('all')
        gc.collect()

        if len(self.clicks) < 3:
            print("⚠ Not enough points clicked (Need 2 Refs + Sites). Skipping.")
            return []

        ref1_pixel = self.clicks[0]
        ref2_pixel = self.clicks[1]
        data_pixels = self.clicks[2:]
        
        print(f"\n✓ Captured {len(self.clicks)} clicks total.")
        print(f"  > Ref 1 Pixel: {int(ref1_pixel[0])}, {int(ref1_pixel[1])}")
        print(f"  > Ref 2 Pixel: {int(ref2_pixel[0])}, {int(ref2_pixel[1])}")
        print(f"  > Site Clicks: {len(data_pixels)}")

        # Now we ask for coordinates (Safe because window is gone)
        print("\n--- ENTER REFERENCE COORDINATES ---")
        refs = []
        for i, (px, py, _) in enumerate([ref1_pixel, ref2_pixel]):
            while True:
                try:
                    sys.stdout.flush()
                    raw = input(f"Paste Lat, Lon for Ref Point {i+1}: ").strip()
                    if not raw: continue
                    
                    raw = raw.replace("(", "").replace(")", "")
                    if "," in raw:
                        parts = raw.split(",")
                    else:
                        parts = raw.split()
                        
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    
                    refs.append({'x': px, 'y': py, 'lat': lat, 'lon': lon})
                    break
                except Exception as e:
                    print(f"  ⚠ Invalid format. Try: 31.25, 34.25")

        # Convert all data points
        new_sites = []
        for i, (px, py, cat) in enumerate(data_pixels):
            lat, lon = self.pixel_to_geo(px, py, refs)
            
            new_sites.append({
                'site_id': f"DIGIT_{len(self.existing_data) + len(new_sites) + 1:05d}",
                'intervention_type': cat,
                'subtype': 'Digitized Location',
                'latitude': round(lat, 5),
                'longitude': round(lon, 5),
                'location_description': f"From Map: {self.image_path.name}",
                'source_map': self.image_path.name
            })
            print(f"  + Processed Site {i+1}: {lat:.5f}, {lon:.5f}")

        return new_sites

    def onclick(self, event):
        if event.xdata is None or event.ydata is None: return
        if event.button != 1: return # Left click only
        
        # Determine what kind of click this is
        idx = len(self.clicks)
        
        if idx == 0:
            # First Reference
            self.ax.plot(event.xdata, event.ydata, 'rx', markersize=15, markeredgewidth=3)
            self.ax.text(event.xdata, event.ydata, " REF 1", color='red', fontsize=12, fontweight='bold')
            cat = 'REF'
        elif idx == 1:
            # Second Reference
            self.ax.plot(event.xdata, event.ydata, 'rx', markersize=15, markeredgewidth=3)
            self.ax.text(event.xdata, event.ydata, " REF 2", color='red', fontsize=12, fontweight='bold')
            cat = 'REF'
        else:
            # Data Point
            self.ax.plot(event.xdata, event.ydata, 'go', markersize=8, markeredgecolor='black', alpha=0.7)
            cat = self.guess_category()
        
        self.clicks.append((event.xdata, event.ydata, cat))
        self.fig.canvas.draw()
        self.update_title()

    def update_title(self):
        count = len(self.clicks)
        if count == 0:
            msg = "STEP 1: Click Reference Point 1 (Top Left)"
            col = 'red'
        elif count == 1:
            msg = "STEP 2: Click Reference Point 2 (Bottom Right)"
            col = 'red'
        else:
            msg = f"STEP 3: Click Data Points (Collected: {count-2}). Close window when done."
            col = 'green'
        
        self.ax.set_title(msg, fontsize=14, color=col, fontweight='bold')
        self.fig.canvas.draw()

    def pixel_to_geo(self, x, y, refs):
        r1, r2 = refs[0], refs[1]
        x_scale = (r2['lon'] - r1['lon']) / (r2['x'] - r1['x'])
        y_scale = (r2['lat'] - r1['lat']) / (r2['y'] - r1['y'])
        lon = r1['lon'] + (x - r1['x']) * x_scale
        lat = r1['lat'] + (y - r1['y']) * y_scale
        return lat, lon
        
    def guess_category(self):
        n = self.image_path.name.lower()
        if 'wash' in n: return 'WASH'
        if 'debris' in n: return 'Debris'
        if 'shelter' in n: return 'Shelter'
        if 'health' in n: return 'Health'
        if 'employment' in n: return 'Employment'
        if 'education' in n: return 'Education'
        return 'Other'

def main():
    if not MAPS_DIR.exists():
        print(f"Folder not found: {MAPS_DIR}")
        return
        
    map_files = sorted([f for f in MAPS_DIR.glob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']])
    
    # Filter out hidden files
    map_files = [f for f in map_files if not f.name.startswith('.')]
    
    if not map_files:
        print("No images found.")
        return
        
    all_sites = []
    
    for f in map_files:
        d = MapDigitizer(f, existing_data=all_sites)
        new_pts = d.run()
        all_sites.extend(new_pts)
        print(f"Finished {f.name}: Collected {len(new_pts)} sites.")

    if all_sites:
        df = pd.DataFrame(all_sites)
        cols = ['site_id', 'intervention_type', 'subtype', 'latitude', 'longitude', 'location_description', 'source_map']
        df = df[cols]
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✓ DONE! Saved {len(df)} sites to {OUTPUT_CSV}")
    else:
        print("\nNo sites collected.")

if __name__ == "__main__":
    main()
