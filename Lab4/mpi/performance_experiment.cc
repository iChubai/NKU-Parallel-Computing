#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

struct ExperimentResult {
    string implementation_name;
    int problem_size;
    double execution_time_ms;
    int num_processes;
    int num_threads;
    bool test_passed;
};

void fRead(int *a, int *b, int *n, int *p, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".in";
    ifstream fin(path);
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}

bool fCheck(int *ab, int n, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".out";
    ifstream fin(path);
    for (int i = 0; i < 2 * n - 1; ++i) {
        int x; fin >> x;
        if (x != ab[i]) {
            return false;
        }
    }
    return true;
}

void runExperiment(const string& executable, const string& name, 
                   vector<ExperimentResult>& results, int rank, int size) {
    vector<int> test_cases = {0, 1, 2, 3};
    
    for (int test_id : test_cases) {
        int a_arr[300000], b_arr[300000];
        int n = 0, p = 0;
        
        if (rank == 0) {
            fRead(a_arr, b_arr, &n, &p, test_id);
        }
        
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(a_arr, n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b_arr, n, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        string command = "mpirun -np " + to_string(size) + " ./" + executable + 
                        " --test-case " + to_string(test_id) + " 2>/dev/null";
        
        int result_code = 0;
        if (rank == 0) {
            result_code = system(command.c_str());
        }
        MPI_Bcast(&result_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            ExperimentResult result;
            result.implementation_name = name;
            result.problem_size = n;
            result.execution_time_ms = (end_time - start_time) * 1000.0;
            result.num_processes = size;
            result.num_threads = omp_get_max_threads();
            result.test_passed = (result_code == 0);
            results.push_back(result);
        }
    }
}

void generateCSVReport(const vector<ExperimentResult>& results, const string& filename) {
    ofstream csv(filename);
    csv << "Implementation,Problem_Size,Execution_Time_ms,Num_Processes,Num_Threads,Test_Passed\n";
    
    for (const auto& result : results) {
        csv << result.implementation_name << ","
            << result.problem_size << ","
            << result.execution_time_ms << ","
            << result.num_processes << ","
            << result.num_threads << ","
            << (result.test_passed ? "True" : "False") << "\n";
    }
    csv.close();
}

void generatePythonVisualization(const string& csv_filename) {
    ofstream py("visualize_results.py");
    py << R"(
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV data
df = pd.read_csv(')" << csv_filename << R"(')

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('NTT MPI Implementation Performance Analysis', fontsize=16, fontweight='bold')

# 1. Execution Time vs Problem Size
ax1 = axes[0, 0]
for impl in df['Implementation'].unique():
    impl_data = df[df['Implementation'] == impl]
    ax1.plot(impl_data['Problem_Size'], impl_data['Execution_Time_ms'], 
             marker='o', linewidth=2, markersize=6, label=impl)

ax1.set_xlabel('Problem Size (n)', fontsize=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Performance vs Problem Size', fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Speedup Analysis
ax2 = axes[0, 1]
baseline_impl = df['Implementation'].iloc[0]
baseline_data = df[df['Implementation'] == baseline_impl]

for impl in df['Implementation'].unique():
    impl_data = df[df['Implementation'] == impl]
    speedup = []
    problem_sizes = []
    
    for size in impl_data['Problem_Size'].unique():
        baseline_time = baseline_data[baseline_data['Problem_Size'] == size]['Execution_Time_ms'].iloc[0]
        impl_time = impl_data[impl_data['Problem_Size'] == size]['Execution_Time_ms'].iloc[0]
        speedup.append(baseline_time / impl_time)
        problem_sizes.append(size)
    
    ax2.plot(problem_sizes, speedup, marker='s', linewidth=2, markersize=6, label=impl)

ax2.set_xlabel('Problem Size (n)', fontsize=12)
ax2.set_ylabel('Speedup Factor', fontsize=12)
ax2.set_title('Speedup Relative to Baseline', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Performance Bar Chart for Large Problem Size
ax3 = axes[1, 0]
large_size_data = df[df['Problem_Size'] == df['Problem_Size'].max()]
implementations = large_size_data['Implementation']
times = large_size_data['Execution_Time_ms']

bars = ax3.bar(range(len(implementations)), times, color=sns.color_palette("husl", len(implementations)))
ax3.set_xlabel('Implementation', fontsize=12)
ax3.set_ylabel('Execution Time (ms)', fontsize=12)
ax3.set_title(f'Performance Comparison (n={df["Problem_Size"].max()})', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(implementations)))
ax3.set_xticklabels(implementations, rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.1f}ms', ha='center', va='bottom', fontsize=10)

# 4. Parallel Efficiency Analysis
ax4 = axes[1, 1]
efficiency_data = []
for impl in df['Implementation'].unique():
    impl_data = df[df['Implementation'] == impl]
    for size in impl_data['Problem_Size'].unique():
        row = impl_data[impl_data['Problem_Size'] == size].iloc[0]
        # Theoretical speedup vs actual speedup
        theoretical_speedup = row['Num_Processes'] * row['Num_Threads']
        # Assume serial time is the baseline implementation time
        serial_time = baseline_data[baseline_data['Problem_Size'] == size]['Execution_Time_ms'].iloc[0]
        actual_speedup = serial_time / row['Execution_Time_ms']
        efficiency = actual_speedup / theoretical_speedup * 100
        efficiency_data.append({
            'Implementation': impl,
            'Problem_Size': size,
            'Efficiency': efficiency,
            'Parallel_Units': theoretical_speedup
        })

eff_df = pd.DataFrame(efficiency_data)
for impl in eff_df['Implementation'].unique():
    impl_eff = eff_df[eff_df['Implementation'] == impl]
    ax4.plot(impl_eff['Problem_Size'], impl_eff['Efficiency'], 
             marker='^', linewidth=2, markersize=6, label=impl)

ax4.set_xlabel('Problem Size (n)', fontsize=12)
ax4.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax4.set_title('Parallel Efficiency Analysis', fontsize=14, fontweight='bold')
ax4.set_xscale('log')
ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal Efficiency')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('ntt_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate summary statistics
print("=== Performance Analysis Summary ===")
print(f"Total implementations tested: {df['Implementation'].nunique()}")
print(f"Problem sizes tested: {sorted(df['Problem_Size'].unique())}")
print(f"All tests passed: {df['Test_Passed'].all()}")

print("\n=== Best Performance by Problem Size ===")
for size in sorted(df['Problem_Size'].unique()):
    size_data = df[df['Problem_Size'] == size]
    best = size_data.loc[size_data['Execution_Time_ms'].idxmin()]
    print(f"n={size}: {best['Implementation']} ({best['Execution_Time_ms']:.2f}ms)")

print("\n=== Average Performance Ranking ===")
avg_performance = df.groupby('Implementation')['Execution_Time_ms'].mean().sort_values()
for i, (impl, avg_time) in enumerate(avg_performance.items(), 1):
    print(f"{i}. {impl}: {avg_time:.2f}ms average")
)";
    py.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "========================================\n";
        cout << "NTT MPI Performance Experiment Framework\n";
        cout << "========================================\n";
        cout << "Processes: " << size << "\n";
        cout << "OpenMP Threads: " << omp_get_max_threads() << "\n";
        cout << "========================================\n\n";
    }

    vector<ExperimentResult> results;
    
    // List of implementations to test
    vector<pair<string, string>> implementations = {
        {"crt_simd_openmp_mpi_final", "CRT+SIMD+OpenMP+MPI"},
        {"radix8_mpi_true_final2", "Radix-8 NTT"},
        {"hybrid_parallel_mpi_fixed", "Hybrid Parallel Strategy"}
    };

    // Run experiments for each implementation
    for (const auto& impl : implementations) {
        if (rank == 0) {
            cout << "Testing: " << impl.second << "\n";
            cout << "----------------------------------------\n";
        }
        
        runExperiment(impl.first, impl.second, results, rank, size);
        
        if (rank == 0) {
            cout << "Completed: " << impl.second << "\n\n";
        }
    }

    // Generate reports and visualizations
    if (rank == 0) {
        cout << "Generating performance reports...\n";
        
        string csv_filename = "ntt_performance_results.csv";
        generateCSVReport(results, csv_filename);
        cout << "CSV report generated: " << csv_filename << "\n";
        
        generatePythonVisualization(csv_filename);
        cout << "Python visualization script generated: visualize_results.py\n";
        
        cout << "\nTo generate visualizations, run:\n";
        cout << "python3 visualize_results.py\n\n";
        
        cout << "Experiment completed successfully!\n";
    }

    MPI_Finalize();
    return 0;
}
