import numpy as np
import struct
import os
import pandas as pd
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

class NCLTLoopClosureSystem:
    def __init__(self, data_dirs, gps_csv_paths, save_in_db, output_file="nclt_descriptors.pkl"):
        # Handle single path or list of paths
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.gps_csv_paths = gps_csv_paths if isinstance(gps_csv_paths, list) else [gps_csv_paths]
        self.output_file = output_file
        self.db_dirs = save_in_db
        
        # Parameters
        self.bins = 100
        self.z_range = (-2.0, 15.0)
        self.scaling = 0.005
        self.offset = -100.0
        self.dist_threshold = 5.0    # Radius for Ground Truth match (Meters)
        self.exclude_recent = 50    # Ignore matches within +/- 100 frames (approx 20-30m buffer)

    # --- PART 1: Processing ---
    def parse_bin_file(self, file_path):
        z_points = []
        record_size = 8
        try:
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(record_size)
                    if not data or len(data) < record_size: break
                    raw = struct.unpack('<HHHBB', data)
                    z_metric = raw[2] * self.scaling + self.offset
                    z_points.append(z_metric)
        except IOError:
            return None
        return np.array(z_points)

    def compute_descriptor(self, z_points):
        if len(z_points) == 0: return np.zeros(self.bins)
        hist, _ = np.histogram(z_points, bins=self.bins, range=self.z_range)
        return hist / (np.sum(hist) + 1e-9)

    def process_data(self):
        if os.path.exists(self.output_file):
            print(f"[INFO] Loading cached data from {self.output_file}...")
            with open(self.output_file, 'rb') as f:
                return pickle.load(f)

        print("[INFO] Processing raw .bin files...")
        
        # 1. Load GPS (Concatenate multiple days)
        print("  Loading GPS...")
        gps_dfs = []
       # Force column names and types to prevent mismatch
        # Based on NCLT format: utime (0), mode (1), num_sats (2), lat (3), lon (4), alt (5), track (6), speed (7)
        col_names = ['utime', 'mode', 'num_sats', 'lat', 'lon', 'alt', 'track', 'speed']
        
        for csv_path in self.gps_csv_paths:
            # Skip header row (if exists), force names
            df = pd.read_csv(csv_path, header=None, names=col_names, skiprows=1)
            
            # Convert lat/lon to float, coerce errors to NaN
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df['utime'] = pd.to_numeric(df['utime'], errors='coerce')
            
            gps_dfs.append(df)
            
        gps_df = pd.concat(gps_dfs, ignore_index=True)
        
        # Drop rows where critical data is NaN
        original_len = len(gps_df)
        gps_df = gps_df.dropna(subset=['lat', 'lon', 'utime'])
        print(f"  [INFO] Dropped {original_len - len(gps_df)} rows with NaN/Invalid data.")
        
        gps_times = gps_df['utime'].values
        lat = gps_df['lat'].values
        lon = gps_df['lon'].values 

        # Lat/Lon to Meters
        R_earth = 6371000
        x_gps = R_earth * lon * np.cos(np.mean(lat))
        y_gps = R_earth * lat
        gps_coords = np.column_stack((x_gps, y_gps))

        # 2. Process Bins

        query_descriptors = []
        query_synced_coords = []
        db_descriptors = []
        db_synced_coords = []

        # print(f"  Extracting from {len(bin_files)} files...")
        for d_ind, data_dir in enumerate(self.data_dirs):
            fnames = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')])
            bin_files = [os.path.join(data_dir, f) for f in fnames]
            save_in_db = self.db_dirs[d_ind]

            for i, filepath in enumerate(tqdm(bin_files)):
                try:
                    # Extract timestamp from filename (works for "1234.bin" or "path/to/1234.bin")
                    fname = os.path.basename(filepath)
                    timestamp = int(os.path.splitext(fname)[0])
                except ValueError:
                    continue

                z_points = self.parse_bin_file(filepath)
                if z_points is not None:
                    desc = self.compute_descriptor(z_points)
                    query_descriptors.append(desc)
                    
                    # Sync GPS
                    idx = np.abs(gps_times - timestamp).argmin()
                    query_synced_coords.append(gps_coords[idx])

                    if save_in_db:
                        db_descriptors.append(desc)
                        db_synced_coords.append(gps_coords[idx])

        results = {
            "query_descriptors": np.array(query_descriptors),
            "query_coords": np.array(query_synced_coords),
            "db_descriptors": np.array(db_descriptors),
            "db_coords": np.array(db_synced_coords)
        }
        
        with open(self.output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"[INFO] Saved to {self.output_file}")
        return results

    # --- PART 2: Memory-Efficient Evaluation ---
    def evaluate_memory_efficient(self, data, batch_size=1000, num_thresholds=200):
        """
        Calculates Precision/Recall without allocating the full NxN matrix.
        Accumulates TP/FP counts for a range of similarity thresholds.
        """
        query_descriptors = data['query_descriptors']
        query_coords = data['query_coords']
        db_descriptors = data['db_descriptors']
        db_coords = data['db_coords']
        query_N = len(query_descriptors)
        db_N = len(db_descriptors)
        
        print(f"\n[INFO] Starting Batched Evaluation on {query_N} query samples with {db_N} db samples.")
        print(f"[INFO] Strategy: Accumulating statistics for {num_thresholds} thresholds.")

        # 1. Define Thresholds for the PR Curve
        # Wasserstein distances are usually small. Range 0.0 to 0.05 covers most cases.
        thresholds = np.linspace(0.01, 0.5, num_thresholds)
        
        # Accumulators for each threshold
        global_TP = np.zeros(num_thresholds)
        global_FP = np.zeros(num_thresholds)
        total_ground_truth_positives = 0

        # Precompute CDFs for speed
        query_CDFs = np.cumsum(query_descriptors, axis=1)
        db_CDFs = np.cumsum(db_descriptors, axis=1)

        # 2. Iterate Batches (Query vs Database)
        # We process the matrix in blocks of (batch_size x batch_size)
        for i in tqdm(range(0, query_N, batch_size)):
            end_i = min(i + batch_size, query_N)
            q_batch_cdf = query_CDFs[i:end_i]         # (B, 100)
            q_batch_coords = query_coords[i:end_i]    # (B, 2)

            # Loop over Database in chunks
            for j in range(0, db_N, batch_size):
                end_j = min(j + batch_size, db_N)
                db_batch_cdf = db_CDFs[j:end_j]      # (B_db, 100)
                db_batch_coords = db_coords[j:end_j] # (B_db, 2)

                # --- A. Compute Wasserstein (Similarity) ---
                # Broadcast: (B, 1, 100) - (1, B_db, 100)
                diff = np.abs(q_batch_cdf[:, np.newaxis, :] - db_batch_cdf[np.newaxis, :, :])
                sim_matrix_batch = np.sum(diff, axis=2) # (B, B_db)

                # --- B. Compute Physical Dist (Ground Truth) ---
                # Broadcast: (B, 1, 2) - (1, B_db, 2)
                pos_diff = q_batch_coords[:, np.newaxis, :] - db_batch_coords[np.newaxis, :, :]
                phys_dist_batch = np.sqrt(np.sum(pos_diff**2, axis=2))

                # --- C. Apply Exclusion Mask ---
                # We need to ignore the diagonal and recent neighbors
                # Indices for Q: [i...end_i], Indices for DB: [j...end_j]
                idx_q = np.arange(i, end_i)
                idx_db = np.arange(j, end_j)
                # Broadcast abs diff: (B, 1) - (1, B_db)
                index_diff = np.abs(idx_q[:, np.newaxis] - idx_db[np.newaxis, :])
                
                # Valid Mask: True if NOT recent
                valid_mask = index_diff > self.exclude_recent

                # --- D. Determine True Matches ---
                # A "True Match" is (Dist < Threshold) AND (Not Recent)
                # print((phys_dist_batch & valid_mask).min())
                gt_batch = (phys_dist_batch < self.dist_threshold) & valid_mask
                
                # Count total existing positives (for Recall calculation)
                total_ground_truth_positives += np.sum(gt_batch)

                # --- E. Accumulate TP/FP for Curve ---
                # We check our sim_matrix against all thresholds at once
                # sim_matrix: (B, B_db). thresholds: (T,)
                # Broadcasting comparison: (B, B_db, 1) < (1, 1, T) -> (B, B_db, T)
                
                # Optimization: To save memory, flatten the batches first
                sim_flat = sim_matrix_batch[valid_mask]  # Only valid pairs
                gt_flat = gt_batch[valid_mask]           # Only valid pairs
                
                if len(sim_flat) > 0:
                    # Shape: (N_valid, 1) < (1, T) -> (N_valid, T) boolean matrix
                    predictions = sim_flat[:, np.newaxis] < thresholds[np.newaxis, :]
                    
                    # True Positives: Predicted True AND Actually True
                    # (N_valid, T) & (N_valid, 1)
                    tp_matrix = predictions & gt_flat[:, np.newaxis]
                    
                    # False Positives: Predicted True AND Actually False
                    fp_matrix = predictions & (~gt_flat[:, np.newaxis])
                    
                    # Sum columns to get counts per threshold
                    global_TP += np.sum(tp_matrix, axis=0)
                    global_FP += np.sum(fp_matrix, axis=0)


        # 3. Finalize Metrics
        if total_ground_truth_positives == 0:
            print("[WARNING] No ground truth loop closures found! Check thresholds/GPS.")
            return

        precision = global_TP / (global_TP + global_FP + 1e-9)
        recall = global_TP / total_ground_truth_positives
        
        # Calculate F1 for all thresholds and find max
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        max_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_idx]
        best_thresh = thresholds[max_f1_idx]

        # Calculate AP (Approximated via Riemann sum of the PR curve)
        # Sort by recall for integration
        sorted_indices = np.argsort(recall)
        r_sorted = recall[sorted_indices]
        p_sorted = precision[sorted_indices]
        ap_score = auc(r_sorted, p_sorted)

        print("-" * 30)
        print(f"Max F1 Score: {max_f1:.4f} (at dist threshold {best_thresh:.4f})")
        print(f"Average Precision (AP): {ap_score:.4f}")
        print(f"Max Precision: {p_sorted.max()}")
        print(f"Total True Loop Closures: {total_ground_truth_positives}")
        print("-" * 30)

        self.plot_pr_curve(recall, precision, ap_score)

    def plot_pr_curve(self, recall, precision, ap):
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {ap:.2f}', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (Batched Evaluation)')
        plt.legend()
        plt.grid(True)
        plt.savefig("nclt_pr_curve.png")
        print("[INFO] Plot saved to nclt_pr_curve.png")
        plt.show()

# --- Entry Point ---
if __name__ == "__main__":
    # Example Lists (Update these)
    DATA_DIRS = [
        "./datasets/2012-01-08/velodyne_sync/", 
        "./datasets/2012-01-15/velodyne_sync/",
        "./datasets/2012-01-22/velodyne_sync/"
    ]
    GPS_CSVS = [
        "./datasets/groundtruth_2012-01-08.csv", 
        "./datasets/groundtruth_2012-01-15.csv",
        "./datasets/groundtruth_2012-01-22.csv"
    ]
    db_dirs = [True, False, False]
    CACHE_FILE = "nclt_full_dataset.pkl"

    system = NCLTLoopClosureSystem(DATA_DIRS, GPS_CSVS, db_dirs, CACHE_FILE)
    
    # 1. Process
    data = system.process_data()
    
    # 2. Evaluate (Memory Efficient)
    # batch_size=500 is safe for 16GB RAM. Increase to 1000 or 2000 if you have 32GB+.
    system.evaluate_memory_efficient(data, batch_size=500, num_thresholds=100)
