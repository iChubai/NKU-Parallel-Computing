import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import io

# Data from ana.md
data_md = """
# main_pthread_v1:
thread == 1 :
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 1.14809 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 441.723 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 444.454 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 430.337 (us) 
thread == 2 :
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 1.11423 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 262.361 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 260.528 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 262.155 (us) 
thread == 3:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 1.22019 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 210.297 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 232.835 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 205.066 (us) 
thread == 4:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 1.52522 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 169.741 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 170.921 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 169.631 (us) 
thread == 8:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 3.03215 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 124.829 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 122.757 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 126.58 (us) 
thread == 16:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 5.68622 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 119.306 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 117.896 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 123.195 (us) 
# main_pthread_v2:
thread == 1 :
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.338087 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 177.339 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 160.093 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 191.006 (us) 
thread == 2 :
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.305287 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 184.039 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 126.154 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 138.714 (us) 
thread == 3:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.45324 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 173.18 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 125.666 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 149.077 (us) 
thread == 4:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.48732 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 114.159 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 118.776 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 123.685 (us) 
thread == 8:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.929653 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 120.11 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 111.225 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 103.549 (us)
thread == 16:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 1.37655 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 102.642 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 96.5295 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 90.5392 (us) 
# main_pthread_v3:
thread == 1 :
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.34111 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 207.166 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 180.6 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 186.195 (us) 
thread == 2 :
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.44275 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 208.719 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 215.832 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 141.917 (us) 
thread == 3:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.547407 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 162.296 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 168.886 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 117.758 (us) 
thread == 4:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.534685 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 122.557 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 159.792 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 140.584 (us) 
thread == 8:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 1.05099 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 112.577 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 100.167 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 108.788 (us) 
thread == 16:
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 1.77188 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 115.053 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 105.634 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 106.924 (us) 
"""

def parse_data(md_content):
    records = []
    current_version = None
    current_threads = None

    for line in io.StringIO(md_content):
        line = line.strip()
        if line.startswith("# main_pthread_"):
            current_version = line.split(":")[0].split("_")[-1] # v1, v2, v3
            if current_version.startswith("v"): # ensure it's v1, v2 or v3
                 current_version = "main_pthread_" + current_version
            else: # for the first entry which might be just # main_pthread
                 current_version = "main_pthread_v1" # Assuming the first block is v1 if not specified
        elif line.startswith("thread =="):
            match = re.search(r"thread == (\d+)", line)
            if match:
                current_threads = int(match.group(1))
        elif line.startswith("average latency for"):
            match = re.search(r"n = (\d+) p = (\d+) : ([\d.]+) \(us\)", line)
            if match and current_version and current_threads:
                n_val = int(match.group(1))
                p_val = int(match.group(2))
                latency = float(match.group(3))
                records.append({
                    "version": current_version,
                    "threads": current_threads,
                    "n": n_val,
                    "p": p_val,
                    "latency": latency
                })
    return pd.DataFrame(records)

df = parse_data(data_md)

# Calculate Speedup and Efficiency
# Speedup is relative to the 1-thread performance of the *same version*
df_list = []
for version in df['version'].unique():
    version_df = df[df['version'] == version].copy()
    # Create a merged key for problem size for easier processing
    version_df['problem_key'] = version_df['n'].astype(str) + '_p' + version_df['p'].astype(str)
    
    baseline_latencies = version_df[version_df['threads'] == 1].set_index('problem_key')['latency']
    
    version_df['baseline_latency'] = version_df['problem_key'].map(baseline_latencies)
    version_df['speedup'] = version_df['baseline_latency'] / version_df['latency']
    version_df['efficiency'] = version_df['speedup'] / version_df['threads']
    df_list.append(version_df)

df_processed = pd.concat(df_list)


# --- Plotting ---
plt.style.use('ggplot') # Using ggplot style for nicer plots

problem_configs = df_processed[['n', 'p']].drop_duplicates().values.tolist()
versions = df_processed['version'].unique()
versions.sort()

# 1. Latency Plots
num_problems = len(problem_configs)
fig1, axes1 = plt.subplots(num_problems, 1, figsize=(12, 6 * num_problems), sharex=True)
if num_problems == 1: axes1 = [axes1] # Make it iterable if only one subplot

for i, (n_val, p_val) in enumerate(problem_configs):
    ax = axes1[i]
    for version in versions:
        subset = df_processed[(df_processed['version'] == version) & (df_processed['n'] == n_val) & (df_processed['p'] == p_val)]
        if not subset.empty:
            ax.plot(subset['threads'], subset['latency'], marker='o', linestyle='-', label=f'{version}')
    ax.set_title(f'Average Latency (n={n_val}, p={p_val})')
    ax.set_ylabel('Latency (µs)')
    ax.legend(title='Implementation Version')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.set_xscale('log', base=2) # Threads on log scale often shows trends better
    ax.set_xticks(df_processed['threads'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())


axes1[-1].set_xlabel('Number of Threads (Log Scale)')
fig1.suptitle('Performance Comparison: Latency vs. Threads', fontsize=16, y=1.02)
fig1.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('ntt_latency_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("Latency comparison plot saved as ntt_latency_comparison.png")


# 2. Speedup Plots
fig2, axes2 = plt.subplots(num_problems, 1, figsize=(12, 6 * num_problems), sharex=True)
if num_problems == 1: axes2 = [axes2]

for i, (n_val, p_val) in enumerate(problem_configs):
    ax = axes2[i]
    unique_threads = sorted(df_processed['threads'].unique())
    ax.plot(unique_threads, unique_threads, linestyle='--', color='gray', label='Ideal Speedup') # Ideal speedup line
    for version in versions:
        subset = df_processed[(df_processed['version'] == version) & (df_processed['n'] == n_val) & (df_processed['p'] == p_val)]
        if not subset.empty:
            ax.plot(subset['threads'], subset['speedup'], marker='o', linestyle='-', label=f'{version}')
    ax.set_title(f'Speedup vs. Threads (n={n_val}, p={p_val})')
    ax.set_ylabel('Speedup (Relative to 1 Thread of Same Version)')
    ax.legend(title='Implementation Version')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks(df_processed['threads'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())


axes2[-1].set_xlabel('Number of Threads (Log Scale)')
fig2.suptitle('Performance Comparison: Speedup vs. Threads', fontsize=16, y=1.02)
fig2.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('ntt_speedup_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("Speedup comparison plot saved as ntt_speedup_comparison.png")

# 3. Efficiency Plots
fig3, axes3 = plt.subplots(num_problems, 1, figsize=(12, 6 * num_problems), sharex=True)
if num_problems == 1: axes3 = [axes3]

for i, (n_val, p_val) in enumerate(problem_configs):
    ax = axes3[i]
    ax.axhline(1.0, linestyle='--', color='gray', label='Ideal Efficiency (100%)') # Ideal efficiency line
    for version in versions:
        subset = df_processed[(df_processed['version'] == version) & (df_processed['n'] == n_val) & (df_processed['p'] == p_val)]
        if not subset.empty:
            ax.plot(subset['threads'], subset['efficiency'], marker='o', linestyle='-', label=f'{version}')
    ax.set_title(f'Parallel Efficiency vs. Threads (n={n_val}, p={p_val})')
    ax.set_ylabel('Efficiency (Speedup / Threads)')
    ax.set_ylim(0, 1.1) # Efficiency typically between 0 and 1
    ax.legend(title='Implementation Version')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks(df_processed['threads'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

axes3[-1].set_xlabel('Number of Threads (Log Scale)')
fig3.suptitle('Performance Comparison: Efficiency vs. Threads', fontsize=16, y=1.02)
fig3.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('ntt_efficiency_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("Efficiency comparison plot saved as ntt_efficiency_comparison.png")

# 4. Latency Grouped Bar Plots
if num_problems > 0 :
    fig4, axes4 = plt.subplots(num_problems, 1, figsize=(14, 7 * num_problems), squeeze=False)
    axes4 = axes4.flatten() # Ensure axes4 is always a flat array for consistent indexing

    for i, (n_val, p_val) in enumerate(problem_configs):
        ax = axes4[i]
        data_for_plot = df_processed[(df_processed['n'] == n_val) & (df_processed['p'] == p_val)]
        
        if not data_for_plot.empty:
            pivot_df = data_for_plot.pivot_table(index='threads', columns='version', values='latency')
            
            if not pivot_df.empty:
                num_versions = len(versions) # Use the globally sorted versions list
                threads_as_str = pivot_df.index.astype(str).tolist()
                x = np.arange(len(threads_as_str))
                total_width_for_group = 0.8
                bar_width = total_width_for_group / num_versions
                
                for j, version_name in enumerate(versions): # Iterate over sorted global versions
                    if version_name in pivot_df.columns: # Check if this version has data for current problem
                        # Calculate offset for each bar in the group
                        offset = (j - num_versions / 2 + 0.5) * bar_width
                        ax.bar(x + offset, pivot_df[version_name], bar_width, label=version_name)

                ax.set_ylabel('Latency (µs)')
                ax.set_title(f'Latency Comparison (Bar) (n={n_val}, p={p_val})')
                ax.set_xticks(x)
                ax.set_xticklabels(threads_as_str)
                ax.legend(title='Implementation Version')
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No data to plot (empty pivot)", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "No data for this configuration", ha='center', va='center')
            
    if num_problems > 0:
      axes4[-1].set_xlabel('Number of Threads')
    fig4.suptitle('Performance Comparison: Latency (Grouped Bar)', fontsize=16, y=1.02 if num_problems > 1 else 1.05)
    fig4.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('ntt_latency_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("Latency grouped bar plot saved as ntt_latency_comparison_bar.png")


# 5. Speedup Grouped Bar Plots
if num_problems > 0:
    fig5, axes5 = plt.subplots(num_problems, 1, figsize=(14, 7 * num_problems), squeeze=False)
    axes5 = axes5.flatten()

    for i, (n_val, p_val) in enumerate(problem_configs):
        ax = axes5[i]
        data_for_plot = df_processed[(df_processed['n'] == n_val) & (df_processed['p'] == p_val)]

        if not data_for_plot.empty:
            pivot_df = data_for_plot.pivot_table(index='threads', columns='version', values='speedup')

            if not pivot_df.empty:
                num_versions = len(versions) # Use the globally sorted versions list
                threads_as_str = pivot_df.index.astype(str).tolist()
                x = np.arange(len(threads_as_str))
                total_width_for_group = 0.8
                bar_width = total_width_for_group / num_versions
                
                for j, version_name in enumerate(versions): # Iterate over sorted global versions
                    if version_name in pivot_df.columns: # Check if this version has data
                        offset = (j - num_versions / 2 + 0.5) * bar_width
                        ax.bar(x + offset, pivot_df[version_name], bar_width, label=version_name)
                
                ax.set_ylabel('Speedup')
                ax.set_title(f'Speedup Comparison (Bar) (n={n_val}, p={p_val})')
                ax.set_xticks(x)
                ax.set_xticklabels(threads_as_str)
                ax.legend(title='Implementation Version')
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No data to plot (empty pivot)", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "No data for this configuration", ha='center', va='center')

    if num_problems > 0:
        axes5[-1].set_xlabel('Number of Threads')
    fig5.suptitle('Performance Comparison: Speedup (Grouped Bar)', fontsize=16, y=1.02 if num_problems > 1 else 1.05)
    fig5.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('ntt_speedup_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("Speedup grouped bar plot saved as ntt_speedup_comparison_bar.png")

print("\\nAll plots generated. Please check for .png files in your current directory.")
print("DataFrame head:\\n", df_processed.head())
print("\\nDataFrame descriptive statistics for latency:\\n", df_processed.groupby(['version', 'n', 'p', 'threads'])['latency'].mean().unstack(level='threads').T.describe())
