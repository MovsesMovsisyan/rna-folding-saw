import random
import numpy as np
    
class Location(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y 
        
    def move(self, dx, dy):
        return Location(self.x + dx, self.y +dy)
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def dist_from(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
    
    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)
    
    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"({self.x}, {self.y})"
    
import random

class StepRule:
    def possible_steps(self):
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def allowed_steps(self, current_location, visited_locations):
        allowed = []

        for dx, dy in self.possible_steps():
            new_location = current_location.move(dx, dy)

            if new_location.y < 0:
                continue

            if new_location in visited_locations:
                continue

            allowed.append((dx, dy))

        return allowed

    def choose_step(self, allowed_steps):
        if not allowed_steps:
            return None
        return random.choice(allowed_steps)

import matplotlib.pylab as plt

class RNAConfiguration:
    def __init__(self, base_location):
        self.positions = [base_location]
        self.visited = {base_location}
        self.step_rule = StepRule()
        
    def current_end(self):
        return self.positions[-1]
    
    def add_step(self):
        current = self.current_end()
        allowed = self.step_rule.allowed_steps(current, self.visited)
        step = self.step_rule.choose_step(allowed)  

        if step is None:
            return False

        new_location = current.move(*step)  
        self.positions.append(new_location)  
        self.visited.add(new_location)

        return True


    
    def walk(self, n):
        for i in range(n):
            if not self.add_step():
                # print('stuck')
                break
            
    def is_stuck(self):
        current = self.current_end()
        allowed = self.step_rule.allowed_steps(current, self.visited)
        return len(allowed) == 0

    def get_positions(self):
        return self.positions

    def end_to_end_distance(self):
        base = self.positions[0]
        end = self.current_end()
        return end.dist_from(base)
    
    
    def plot(self, contacts=None):
        positions = self.get_positions()
        x_coords = [loc.x for loc in positions]
        y_coords = [loc.y for loc in positions]

        plt.figure(figsize=(8,8))
        
        # 1. Plot the main walk path (The "Backbone")
        plt.plot(x_coords, y_coords, '-', color='black', alpha=0.5, label='RNA Backbone', zorder=1)
        
        # 2. Plot all points with a single, neutral color
        plt.scatter(x_coords, y_coords, color='blue', s=40, label='Monomers', zorder=2)

        # 3. Visualize the matching contacts (Red dashed lines)
        if contacts:
            for i, j in contacts:
                p1, p2 = positions[i], positions[j]
                plt.plot([p1.x, p2.x], [p1.y, p2.y], 'r--', lw=2, alpha=0.9, zorder=3)

        # 4. Clearly mark the start and end
        plt.scatter(x_coords[0], y_coords[0], color='green', s=120, label='Start (0,0)', edgecolors='black', zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], color='red', s=120, label='End Point', edgecolors='black', zorder=5)

        plt.title(f"RNA Self-Avoiding Walk\nSteps: {len(positions)-1} | Contacts: {len(contacts) if contacts else 0}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.axis('equal') # This prevents the grid from looking stretched
        plt.show()
        
    def matching_contacts(self):
        positions = self.get_positions()
        # Create a lookup for O(1) coordinate-to-index checks
        pos_dict = {(p.x, p.y): i for i, p in enumerate(positions)}

        n = len(positions)
        matched = [False] * n
        contacts = []

        for i, p in enumerate(positions):
            # Skip if this monomer already found a partner in a previous iteration
            if matched[i]:
                continue

            x, y = p.x, p.y
            
            # 1. We reset the "best candidate" search for EACH monomer i
            closest_index = 999999999999999
            target_j = None

            # 2. Look at all 4 neighbors to find the absolute smallest index
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if (nx, ny) in pos_dict:
                    j = pos_dict[(nx, ny)]
                    
                    # Logic: Must not be backbone (abs > 1) AND not already matched
                    if abs(i - j) > 1 and not matched[j]:
                        if j < closest_index:
                            closest_index = j
                            target_j = j

            # 3. Only AFTER checking all 4 neighbors, if we found a valid j, we match
            if target_j is not None:
                matched[i] = True
                matched[target_j] = True
                contacts.append((i, target_j))
                # Note: No 'break' needed here because we finished the neighbor loop
                
        return contacts
    


    def matching_matrix(self, contacts):
        n = len(self.positions)

        M = [[0]*n for _ in range(n)]

        for i, j in contacts:
            M[i][j] = 1
            M[j][i] = 1

        return M
    
    def l(self, matrix):
        N = len(self.get_positions())
        l = 0
        for i in range(N):
            for j in range(i - 1):
                for k in range (i, N):
                    l += matrix[j][k]
        return 2 * l / (N * (N + 1))
  
            
    def D(self):
        positions = self.get_positions()
        N = len(positions)
        y_dist = 0
        for p in positions:
            y_dist += p.y
        return 2 * y_dist / (N * (N + 1))

    
    
    
    
    
    
from multiprocessing import Pool, cpu_count
import random

def single_run1(max_steps):
    base = Location(0, 0)
    polymer = RNAConfiguration(base)
    polymer.walk(max_steps)

    length = len(polymer.get_positions()) - 1
    dist = polymer.end_to_end_distance()

    return length, dist


def run_experiment_parallel(num_runs=10000, max_steps=1000):
    num_workers = cpu_count()

    print(f"Using {num_workers} cores...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(single_run1, [max_steps] * num_runs)

    lengths = [r[0] for r in results]
    distances = [r[1] for r in results]

    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)
    avg_distance = sum(distances) / len(distances)

    print("===== RESULTS =====")
    print(f"Runs: {num_runs}")
    print(f"Average length: {avg_length}")
    print(f"Min length: {min_length}")
    print(f"Max length: {max_length}")
    print(f"Average end-to-end distance: {avg_distance}")

    import matplotlib.pyplot as plt
    plt.hist(lengths, bins=30)
    plt.title("Distribution of Walk Lengths")
    plt.xlabel("Length before getting stuck")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()



def print_matrix(M):
    print("\nConnection Matrix:\n")
    for row in M:
        print(" ".join(f"{val:2d}" for val in row))



def single_run2(max_steps):
    # base = Location(0, 0)
    # polymer = RNAConfiguration(base)
    # polymer.walk(max_steps)

    # # 1. We must calculate contacts and the matrix to get 'l'
    # contacts = polymer.matching_contacts()
    # matrix = polymer.matching_matrix(contacts)

    # # 2. Extract the metrics
    # l_val = polymer.l(matrix)
    # D_val = polymer.D()
    # actual_length = len(polymer.get_positions()) - 1

    # return l_val, D_val, actual_length # Returns 3 values

    base = Location(0, 0)
    polymer = RNAConfiguration(base)
    polymer.walk(max_steps)
    length_threshold = 100

    actual_length = len(polymer.get_positions()) - 1

    # EARLY EXIT: If the walk didn't reach our threshold, 
    # return None immediately and skip the heavy math.
    if actual_length < length_threshold:
        return None

    # Only perform these operations for "successful" walks
    contacts = polymer.matching_contacts()
    matrix = polymer.matching_matrix(contacts)

    l_val = polymer.l(matrix)
    D_val = polymer.D()

    return l_val, D_val, actual_length

def run_comparison_experiment(num_runs, max_steps):
    num_workers = cpu_count()
    print(f"Simulating {num_runs} walks on {num_workers} cores...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(single_run2, [max_steps] * num_runs)

    filtered_results = [r for r in results if r is not None]
    
    if not filtered_results:
        print("No walks reached the length threshold.")
        return

    l_values = np.array([r[0] for r in filtered_results])
    D_values = np.array([r[1] for r in filtered_results])
    lengths  = np.array([r[2] for r in filtered_results])

    # Use D / (1 - l): brush height per *free* (unpaired) monomer.
    # This is the physically correct stretch factor; D/l would divide the
    # brush extension by the contact density, which is not what determines
    # how stretched the chain is.
    valid = (l_values < 1.0) & np.isfinite(l_values) & np.isfinite(D_values)
    dl_ratios = D_values[valid] / (1.0 - l_values[valid])
    n_target = 200
    if len(dl_ratios) >= n_target:
        idx = np.random.choice(len(dl_ratios), size=n_target, replace=False)
        dl_save = dl_ratios[idx]
    else:
        dl_save = dl_ratios
    import os
    os.makedirs("datafiles", exist_ok=True)
    np.save("datafiles/dl_ratios.npy", dl_save)
    print(f"Saved {len(dl_save)} D/l ratios to datafiles/dl_ratios.npy")

    # --- PLOT 1: RAW SCATTER DATA ---
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(l_values, D_values, c=lengths, cmap='viridis', alpha=0.5, s=5)
    plt.colorbar(scatter, label='Walk Length')
    plt.title("Correlation: l vs. D (Raw Data)")
    plt.xlabel("l (Internal Density)")
    plt.ylabel("D (Average Height)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

    # --- CALCULATE BINNED MEANS ---
    dx = 0.02 # Width of each interval
    bins = np.arange(0, 1 + dx, dx)
    midpoints = []
    means = []

    for i in range(len(bins) - 1):
        mask = (l_values >= bins[i]) & (l_values < bins[i+1])
        if np.any(mask):
            midpoints.append(bins[i] + dx/2)
            means.append(np.mean(D_values[mask]))

    midpoints = np.array(midpoints)
    means = np.array(means)

    # --- PLOT 2: BINNED MEANS & INTERPOLATION (Polyfit) ---
    plt.figure(figsize=(12, 8))
    
    # Plot the binned points we just calculated
    plt.plot(midpoints, means, 'o', color='black', label='Binned Means (dx=0.02)')

    # Perform Polyfit Interpolation on those means
    degree = 3
    coeffs = np.polyfit(midpoints, means, degree)
    poly_func = np.poly1d(coeffs)
    
    x_smooth = np.linspace(midpoints.min(), midpoints.max(), 200)
    plt.plot(x_smooth, poly_func(x_smooth), 'r-', lw=2, label=f'Cubic Interpolation (Degree {degree})')

    plt.title("Binned Average Trend: l vs. D")
    plt.xlabel("l (Internal Density)")
    plt.ylabel("Mean D (Average Height)")
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    # Summary Stats
    print("\n--- Statistical Summary ---")
    print(f"Mean l (Internal Density): {np.mean(l_values):.4f}")
    print(f"Mean D (Avg Height):       {np.mean(D_values):.4f}")
    print(f"Number of points:          {len(filtered_results)}")


if __name__ == "__main__":
    run_comparison_experiment(num_runs=100000, max_steps=1000)
