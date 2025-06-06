#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;
using u32 = uint32_t;
using i64 = long long;

void fRead(int *a,int *b,int *n,int *p,int id){
    string path="/nttdata/"+to_string(id)+".in";
    ifstream fin(path); fin>>*n>>*p;
    for(int i=0;i<*n;i++) fin>>a[i];
    for(int i=0;i<*n;i++) fin>>b[i];
}
void fCheck(int *ab,int n,int id){
    string path="/nttdata/"+to_string(id)+".out";
    ifstream fin(path);
    bool correct = true;
    for(int i=0;i<2*n-1;i++){
        int x;
        fin>>x;
        if(x!=ab[i]){
            cout<<"多项式乘法结果错误 (id="<<id<<") at position "<<i<<": expected "<<x<<", got "<<ab[i]<<"\n";
            cout<<"Complete result: ";
            for(int j=0; j<2*n-1; j++) cout<<ab[j]<<" ";
            cout<<"\n";
            correct = false;
            break;
        }
    }
    if(correct) cout<<"多项式乘法结果正确 (id="<<id<<")\n";
}
void fWrite(int *ab,int n,int id){
    string path="files/"+to_string(id)+".out";
    ofstream fout(path); for(int i=0;i<2*n-1;i++) fout<<ab[i]<<'\n';
}

static inline int mod_pow(i64 x,i64 y,int p){
    i64 r=1%p; x%=p; while(y){ if(y&1) r=r*x%p; x=x*x%p; y>>=1;} return int(r);
}

static void bitrev(int *a,int n){
    int lg=__builtin_ctz(n);
    for(int i=0;i<n;++i){
        int rev=0;
        for(int j=0;j<lg;++j) if(i>>j&1) rev|=1<<(lg-1-j);
        if(i<rev) swap(a[i],a[rev]);
    }
}
static void digrev4(int *a,int n){
    int pairs=__builtin_ctz(n)>>1;
    for(int i=0;i<n;++i){
        int rev=0,t=i;
        for(int j=0;j<pairs;++j){ rev=(rev<<2)|(t&3); t>>=2; }
        if(i<rev) swap(a[i],a[rev]);
    }
}

static void radix2_layer_serial(int *a,int n,bool inv,int p, int rank, int id){
    int h=n>>1;
    int wn=mod_pow(3,(p-1)/n,p);
    if(inv) wn=mod_pow(wn,p-2,p);
    int w=1;

    for(int j=0;j<h;++j){
        int u=a[j];
        int v=1LL*a[j+h]*w%p;
        a[j]   =(u+v)%p;
        a[j+h] =(u-v+p)%p;
        w=1LL*w*wn%p;
    }
}

static void radix4_layers_mpi(int *a, int n, bool inv, int p,
                              int rank, int size, MPI_Comm comm, int id_for_print)
{
    bool odd = __builtin_ctz(n) & 1;
    int start_len = odd ? 8 : 4;

    for (int len = start_len; len <= n; len <<= 2) {
        int m = len >> 2;

        if (m <= 0) {
            continue;
        }

        int wn_base = mod_pow(3, (p - 1) / len, p);
        if (inv) {
            wn_base = mod_pow(wn_base, p - 2, p);
        }
        int J_factor = mod_pow(wn_base, m, p);

        vector<int> w_pow_j(m);
        w_pow_j[0] = 1;
        for (int j_idx = 1; j_idx < m; ++j_idx) {
            w_pow_j[j_idx] = (1LL * w_pow_j[j_idx - 1] * wn_base) % p;
        }

        int totalRows = n / len;
        if (totalRows <= 0) {
            continue;
        }

        int base_rows_per_proc = totalRows / size;
        int rem_rows = totalRows % size;
        int myRows = base_rows_per_proc + (rank < rem_rows ? 1 : 0);
        int rowStartOffset = rank * base_rows_per_proc + min(rank, rem_rows);

        for (int r_idx = 0; r_idx < myRows; ++r_idx) {
            int blk_offset = (rowStartOffset + r_idx) * len;

            if (blk_offset + len > n) {
                break;
            }

            for (int j_group_idx = 0; j_group_idx < m; ++j_group_idx) {
                if (blk_offset + j_group_idx + 3 * m >= n) {
                    break;
                }

                long long current_w_j = w_pow_j[j_group_idx];
                long long current_w_2j = (current_w_j * current_w_j) % p;
                long long current_w_3j = (current_w_2j * current_w_j) % p;

                int termA_in = a[blk_offset + j_group_idx];
                int termB_in = a[blk_offset + j_group_idx + m];
                int termC_in = a[blk_offset + j_group_idx + 2 * m];
                int termD_in = a[blk_offset + j_group_idx + 3 * m];

                int A_tw = termA_in;
                int B_tw = (1LL * termB_in * current_w_j) % p;
                int C_tw = (1LL * termC_in * current_w_2j) % p;
                int D_tw = (1LL * termD_in * current_w_3j) % p;

                int T0 = (A_tw + C_tw) % p;
                int T1 = (A_tw + p - C_tw) % p;
                int T2 = (B_tw + D_tw) % p;
                int T3 = (1LL * (B_tw + p - D_tw) * J_factor) % p;

                a[blk_offset + j_group_idx]        = (T0 + T2) % p;
                a[blk_offset + j_group_idx + m]    = (T1 + T3) % p;
                a[blk_offset + j_group_idx + 2*m]  = (T0 + p - T2) % p;
                a[blk_offset + j_group_idx + 3*m]  = (T1 + p - T3) % p;
            }
        }

        vector<int> recvcounts(size);
        vector<int> displs(size);
        for (int r_proc = 0; r_proc < size; ++r_proc) {
            int rows_for_r_proc = base_rows_per_proc + (r_proc < rem_rows ? 1 : 0);
            recvcounts[r_proc] = rows_for_r_proc * len;
            displs[r_proc] = (r_proc * base_rows_per_proc + min(r_proc, rem_rows)) * len;
        }

        int sendcnt_local = myRows * len;

        if (sendcnt_local > 0 && rowStartOffset * len < n) {
            int actual_copy_size = min(sendcnt_local, n - rowStartOffset * len);
            vector<int> sendbuf_local_storage(sendcnt_local, 0);

            if (actual_copy_size > 0) {
                memcpy(sendbuf_local_storage.data(), a + rowStartOffset * len,
                       sizeof(int) * actual_copy_size);
            }

            MPI_Allgatherv(sendbuf_local_storage.data(),
                           sendcnt_local,
                           MPI_INT,
                           a,
                           recvcounts.data(),
                           displs.data(),
                           MPI_INT,
                           comm);
        } else {
            MPI_Allgatherv(nullptr,
                           0,
                           MPI_INT,
                           a,
                           recvcounts.data(),
                           displs.data(),
                           MPI_INT,
                           comm);
        }
    }
}

void ntt_mpi(int *a,int n,bool inv,int p,int rank,int size,MPI_Comm comm, int id_for_print){
    if (n <= 16) {
        if (rank == 0) {
            int lg = __builtin_ctz(n);
            for (int i = 0; i < n; ++i) {
                int rev = 0;
                for (int j = 0; j < lg; ++j) {
                    if (i >> j & 1) rev |= 1 << (lg - 1 - j);
                }
                if (i < rev) swap(a[i], a[rev]);
            }

            for (int len = 2; len <= n; len <<= 1) {
                int wn = mod_pow(3, (p - 1) / len, p);
                if (inv) wn = mod_pow(wn, p - 2, p);

                for (int i = 0; i < n; i += len) {
                    int w = 1;
                    for (int j = 0; j < len / 2; ++j) {
                        int u = a[i + j];
                        int v = 1LL * a[i + j + len / 2] * w % p;
                        a[i + j] = (u + v) % p;
                        a[i + j + len / 2] = (u - v + p) % p;
                        w = 1LL * w * wn % p;
                    }
                }
            }

            if (inv) {
                int invN = mod_pow(n, p - 2, p);
                for (int i = 0; i < n; ++i) {
                    a[i] = 1LL * a[i] * invN % p;
                }
            }
        }
        MPI_Bcast(a, n, MPI_INT, 0, comm);
        return;
    }

    bool odd = __builtin_ctz(n) & 1;

    if(rank == 0){
        if(odd){
            bitrev(a, n);
            radix2_layer_serial(a, n, inv, p, rank, id_for_print);
        } else {
            digrev4(a, n);
        }
    }
    MPI_Bcast(a, n, MPI_INT, 0, comm);

    radix4_layers_mpi(a, n, inv, p, rank, size, comm, id_for_print);

    if(inv){
        if(rank == 0){
            int invN = mod_pow(n, p - 2, p);
            for(int i = 0; i < n; ++i) a[i] = 1LL * a[i] * invN % p;
        }
        MPI_Bcast(a, n, MPI_INT, 0, comm);
    }
}

void poly_mul_mpi(const int *A_const, const int *B_const, int *C, int n_orig, int p,
                  int rank, int size, MPI_Comm comm, int current_id_for_print)
{
    if (n_orig <= 0) {
        if (rank == 0) {
            std::cerr << "Error: n_orig must be positive, got " << n_orig << std::endl;
        }
        return;
    }

    int N = 1;
    while (N < 2 * n_orig) N <<= 1;

    if (N <= 0 || N > 1000000) {
        if (rank == 0) {
            std::cerr << "Error: Invalid N = " << N << " for n_orig = " << n_orig << std::endl;
        }
        return;
    }

    vector<int> pa(N), pb(N);

    for (int i = 0; i < n_orig; ++i) {
        pa[i] = ((A_const[i] % p) + p) % p;
        pb[i] = ((B_const[i] % p) + p) % p;
    }
    for (int i = n_orig; i < N; ++i) {
        pa[i] = 0;
        pb[i] = 0;
    }

    ntt_mpi(pa.data(), N, false, p, rank, size, comm, current_id_for_print);
    ntt_mpi(pb.data(), N, false, p, rank, size, comm, current_id_for_print);

    int items_per_rank_base = N / size;
    int remainder_items = N % size;

    int my_item_count = items_per_rank_base + (rank < remainder_items ? 1 : 0);
    int my_start_index_in_global_N = rank * items_per_rank_base + min(rank, remainder_items);

    if (my_start_index_in_global_N >= N) {
        my_item_count = 0;
        my_start_index_in_global_N = 0;
    } else if (my_start_index_in_global_N + my_item_count > N) {
        my_item_count = N - my_start_index_in_global_N;
    }

    vector<int> local_product_block(max(my_item_count, 1));
    for (int i = 0; i < my_item_count; ++i) {
        int global_idx = my_start_index_in_global_N + i;
        if (global_idx < N) {
            local_product_block[i] = (1LL * pa[global_idx] * pb[global_idx]) % p;
        } else {
            local_product_block[i] = 0;
        }
    }

    vector<int> recvcounts(size);
    vector<int> displs(size);
    int current_displ = 0;
    for (int r_proc = 0; r_proc < size; ++r_proc) {
        recvcounts[r_proc] = items_per_rank_base + (r_proc < remainder_items ? 1 : 0);
        displs[r_proc] = current_displ;
        current_displ += recvcounts[r_proc];
    }

    MPI_Allgatherv(local_product_block.data(),
                   my_item_count, MPI_INT,
                   pa.data(),
                   recvcounts.data(), displs.data(),
                   MPI_INT, comm);

    ntt_mpi(pa.data(), N, true, p, rank, size, comm, current_id_for_print);

    if (rank == 0) {
        int result_size = 2 * n_orig - 1;
        if (result_size > 0 && result_size <= N) {
            memcpy(C, pa.data(), sizeof(int) * result_size);
        } else {
            std::cerr << "Error: Invalid result size " << result_size << std::endl;
        }
    }
}

void print_array(const char* name, const int* arr, int n, int N_total, int rank_to_print, int current_rank, int id_to_print, int current_id, const char* stage) {
    if (current_rank == rank_to_print && current_id == id_to_print) {
        std::cout << "Rank " << current_rank << ", ID " << current_id << ", Stage: " << stage << " --- " << name << " (first " << n << " of " << N_total << " elements): ";
        for (int i = 0; i < n; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc,char** argv){
    int mpi_init_result = MPI_Init(&argc, &argv);
    if (mpi_init_result != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed with error code: " << mpi_init_result << std::endl;
        return 1;
    }

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> a(300000, 0), b(300000, 0), ab(600000, 0);

    for(int id = 0; id <= 3; ++id){
        int n = 0, p = 0;

        if(rank == 0) {
            try {
                fRead(a.data(), b.data(), &n, &p, id);

                if (n <= 0 || n > 300000) {
                    std::cerr << "Error: Invalid n = " << n << " for id = " << id << std::endl;
                    n = 0;
                }
                if (p <= 1) {
                    std::cerr << "Error: Invalid p = " << p << " for id = " << id << std::endl;
                    p = 0;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error reading input for id " << id << ": " << e.what() << std::endl;
                n = 0;
                p = 0;
            }
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (n <= 0 || p <= 1) {
            if (rank == 0) {
                std::cerr << "Skipping test case id=" << id << " due to invalid parameters" << std::endl;
            }
            continue;
        }

        MPI_Bcast(a.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        poly_mul_mpi(a.data(), b.data(), ab.data(), n, p, rank, size, MPI_COMM_WORLD, id);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if(rank == 0){
            fCheck(ab.data(), n, id);
            cout << "id=" << id << " n=" << n << " p=" << p
                 << " procs=" << size << " time=" << (t1-t0)*1e6 << " us\n";
        }
    }

    MPI_Finalize();
    return 0;
}
