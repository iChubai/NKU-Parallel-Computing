#include <bits/stdc++.h>
using namespace std;

using u64 = uint64_t;
using u32 = uint32_t;

const int MODS[2] = {7340033, 104857601};
const int PRIMITIVE_ROOT = 3;
const u64 BIG_MOD = 7696582450348003ULL; 

template<typename T>
T modpow(T a, T b, T mod) {
    T res = 1;
    while (b) {
        if (b & 1) res = (unsigned __int128)res * a % mod;
        a = (unsigned __int128)a * a % mod;
        b >>= 1;
    }
    return res;
}

template<typename T>
void prepare_roots(vector<T>& roots, int n, T mod, T primitive_root, bool invert) {
    T g = modpow(primitive_root, (mod - 1) / n, mod);
    if (invert) g = modpow(g, mod - 2, mod);
    roots.resize(n);
    roots[0] = 1;
    for (int i = 1; i < n; ++i) {
        roots[i] = (unsigned __int128)roots[i - 1] * g % mod;
    }
}

template<typename T>
void ntt(vector<T>& a, bool invert, T mod, T primitive_root) {
    int n = a.size();
    vector<T> roots;
    prepare_roots(roots, n, mod, primitive_root, invert);
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; ++j) {
                T u = a[i + j];
                T v = (unsigned __int128)a[i + j + half] * roots[step * j] % mod;
                T t = u + v;
                if (t >= mod) t -= mod;
                T t2 = u + mod - v;
                if (t2 >= mod) t2 -= mod;
                a[i + j] = t;
                a[i + j + half] = t2;
            }
        }
    }
    if (invert) {
        T inv_n = modpow((T)n, mod - 2, mod);
        for (int i = 0; i < n; ++i) {
            a[i] = (unsigned __int128)a[i] * inv_n % mod;
        }
    }
}

u64 crt2(u64 r1, u64 r2, u64 m1, u64 m2) {
    u64 m1_inv_m2 = modpow(m1, m2 - 2, m2);
    u64 x1 = r1;
    u64 x2 = ((r2 + m2 - x1 % m2) * m1_inv_m2) % m2;
    return (x1 + m1 * x2) % (m1 * m2);
}

template<typename T>
vector<T> convolve(const vector<T>& a, int s1, const vector<T>& b, int s2, T mod, T primitive_root, bool cyclic = false) {
    int s = cyclic ? max(s1, s2) : s1 + s2 - 1;
    int size = 1; while (size < 2 * s - 1) size <<= 1;
    vector<T> c = a, d = b;
    c.resize(size); d.resize(size);
    ntt(c, false, mod, primitive_root);
    ntt(d, false, mod, primitive_root);
    for (int i = 0; i < size; ++i) c[i] = (unsigned __int128)c[i] * d[i] % mod;
    ntt(c, true, mod, primitive_root);
    if (cyclic) {
        for (int i = s; i < size; ++i) c[i % s] = (c[i % s] + c[i]) % mod;
        c.resize(s);
    } else {
        c.resize(s1 + s2 - 1);
    }
    return c;
}

void fRead(u64 *a, u64 *b, int *n, u64 *p, int input_id) {
    string str1 = "/nttdata/";
    string str2 = to_string(input_id);
    string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; i++) {
        fin >> b[i];
    }
}

void fCheck(const u64 *ab, int n, int input_id) {
    string str1 = "/nttdata/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    for (int i = 0; i < n * 2 - 1; i++) {
        u64 x;
        fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误" << endl;
            return;
        }
    }
    cout << "多项式乘法结果正确" << endl;
    return;
}

void fWrite(const u64 *ab, int n, int input_id) {
    string str1 = "files/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    ofstream fout;
    fout.open(output_path, ios::out);
    for (int i = 0; i < n * 2 - 1; i++) {
        fout << ab[i] << '\n';
    }
}

// 全局数组
u64 a[300000], b[300000], result[300000];

int main() {
    try {
        const int test_begin = 0;
        const int test_end = 4;
        
        for (int i = test_begin; i <= test_end; ++i) {
            int n;
            u64 mod;
            
            fRead(a, b, &n, &mod, i);
            memset(result, 0, sizeof(result));
            
            auto start = chrono::high_resolution_clock::now();
            
            if (mod == BIG_MOD) {
                int len = 2 * n + 1;
                vector<vector<u64>> results(2, vector<u64>(len));
                
                for (int t = 0; t < 2; ++t) {
                    u32 current_mod = MODS[t];
                    vector<u32> aa(n + 1), bb(n + 1);
                    for (int i = 0; i <= n; ++i) {
                        aa[i] = a[i];
                        bb[i] = b[i];
                    }
                    
                    vector<u32> res = convolve(aa, n + 1, bb, n + 1, 
                                             current_mod, (u32)PRIMITIVE_ROOT);
                    for (int i = 0; i < len; ++i) {
                        results[t][i] = res[i];
                    }
                }
                
                for (int i = 0; i < len; ++i) {
                    result[i] = crt2(results[0][i], results[1][i], 
                                   MODS[0], MODS[1]) % mod;
                }
            } else if (mod > u32(-1)) {
                vector<u64> aa(n + 1), bb(n + 1);
                for (int i = 0; i <= n; ++i) {
                    aa[i] = a[i];
                    bb[i] = b[i];
                }
                
                vector<u64> res = convolve(aa, n + 1, bb, n + 1, 
                                         mod, (u64)PRIMITIVE_ROOT);
                for (int i = 0; i <= 2 * n; ++i) {
                    result[i] = res[i];
                }
            } else {
                vector<u32> aa(n + 1), bb(n + 1);
                for (int i = 0; i <= n; ++i) {
                    aa[i] = a[i];
                    bb[i] = b[i];
                }
                
                vector<u32> res = convolve(aa, n + 1, bb, n + 1, 
                                         (u32)mod, (u32)PRIMITIVE_ROOT);
                for (int i = 0; i <= 2 * n; ++i) {
                    result[i] = res[i];
                }
            }
            
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double, ratio<1, 1000>> elapsed = end - start;
            double execution_time = elapsed.count();
            
            fCheck(result, n, i);
            cout << "average latency for n = " << n << " p = " << mod << " : " 
                 << execution_time << " (us)" << endl;
            
            fWrite(result, n, i);
        }
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
        return 1;
    }
    
    return 0;
} 