#include <stdint.h>
#include <type_traits>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

using u32 = uint32_t;
using u64 = uint64_t;
using u128 = __uint128_t;

using t_widen_default = u128;
template <typename T>
using t_widen = typename std::conditional<std::is_same<T, u32>::value, u64,
                                          typename std::conditional<std::is_same<T, u64>::value, u128, t_widen_default>::type>::type;

// === Generic Modulo Operations ===
template <typename T>
class ModGeneric
{
  using T2 = t_widen<T>;

public:
  ModGeneric(T _mod) : mod(_mod) {}
  ModGeneric(const ModGeneric &) = delete;
  ModGeneric &operator=(const ModGeneric &) = delete;

  T add(T a, T b) const { return (a + b) % mod; }
  T sub(T a, T b) const { return (a >= b) ? (a - b) : (a + mod - b); }
  T mul(T a, T b) const { return ((T2)a * (T2)b) % mod; }
  T pow(T base, T exp) const
  {
    T result = 1;
    while (exp > 0)
    {
      if (exp & 1)
      {
        result = mul(result, base);
      }
      base = mul(base, base);
      exp >>= 1;
    }
    return result;
  }
  T inv(T x) const { return pow(x, mod - 2); }

private:
  T mod;
};

static const u64 CRT_MODS[] = {998244353, 1004535809, 469762049};
static const u64 CRT_NUMS = 3;

static const u64 SPECIAL_BIG_MOD_P_VAL = 1337006139375617ULL; 
static const u64 SPECIAL_CRT_M1 = 7340033;
static const u64 SPECIAL_CRT_M2 = 104857601;

inline u64 expand_n(u64 n) {
    u64 lg_n = 0;
    while ((1u << lg_n) < n) ++lg_n;
    return 1 << lg_n;
}

inline void CRT_combine(u128 *ab, u64 **ab_crt, u64 n) {
    u128 m = CRT_MODS[0];
    for (u64 i = 1; i < CRT_NUMS; ++i) {
        ModGeneric<u64> mod_op(CRT_MODS[i]); // Use ModGeneric
        u128 inv = mod_op.inv(m % CRT_MODS[i]);
        for (u64 j = 0; j < n; j++) {
            u128 x = ab[j];
            u64 t = mod_op.sub(ab_crt[i][j], x % CRT_MODS[i]);
            ab[j] = x + m * mod_op.mul(t, inv);
        }
        m *= CRT_MODS[i];
    }
}

inline void CRT_combine_2(u128 *ab, u64 **ab_crt, u64 n) {
    for (u64 i = 0; i < n; ++i) {
        u128 x = ab_crt[0][i], m = CRT_MODS[0];
        for (u64 j = 1; j < CRT_NUMS; ++j) {
            ModGeneric<u64> mod_op(CRT_MODS[j]); // Use ModGeneric
            u64 t = mod_op.sub(ab_crt[j][i], x % CRT_MODS[j]);
            x += m * mod_op.mul(t, mod_op.inv(m % CRT_MODS[j]));
            m *= CRT_MODS[j];
        }
        ab[i] = x;
    }
}

// 双模CRT合并函数 (类似 main_big_p.cc 中的 crt2)
// 计算结果 X 满足: X = r1 (mod m1), X = r2 (mod m2)
// 返回值是 X mod (m1*m2)，且 0 <= X < m1*m2
inline u64 crt_combine_special_two(u64 r1, u64 m1, u64 r2, u64 m2) {
    ModGeneric<u64> mod_m2(m2); // Use ModGeneric for operations modulo m2
    // 计算 m1_inv_m2 = m1^(m2-2) mod m2
    u64 m1_for_inv = m1 % m2;
    u64 m1_inv_m2 = mod_m2.pow(m1_for_inv, m2 - 2);

    u64 x1 = r1;
    // x2 = ((r2 - (x1 % m2) + m2) % m2 * m1_inv_m2) % m2
    u64 r1_mod_m2 = x1 % m2;
    
    u64 term_val = r2;
    if (term_val < r1_mod_m2) { // Ensure (r2 - r1_mod_m2) is non-negative before modulo
        term_val += m2;
    }
    term_val -= r1_mod_m2; // Now term_val is (r2 - r1_mod_m2) or (r2 - r1_mod_m2 + m2)

    // term_val is now equivalent to (r2 - (x1 % m2)) mod m2
    u64 x2 = mod_m2.mul(term_val, m1_inv_m2);

    // 结果是 x1 + m1 * x2. 这个值应该小于 m1*m2.
    // 为了确保结果在 [0, m1*m2 - 1] 范围内，显式取模 (m1*m2)
    u128 combined_u128 = (u128)x1 + (u128)m1 * x2;
    return combined_u128 % ((u128)m1 * m2);
}

// 前向声明 poly_multiply_ntt
void poly_multiply_ntt(u64 *a, u64 *b, u64 *ab, u64 n, u64 p_ntt);

// 使用三模CRT的多项式乘法示例函数 (修改后)
inline void poly_multiply_crt(u64 *a, u64 *b, u64 *ab, u64 n, u64 p) {
    u64 n_expanded = expand_n(2 * n - 1);

    if (p == SPECIAL_BIG_MOD_P_VAL) {
        // 针对特定超大模数p，使用双模CRT策略 (类似main_big_p.cc)
        u64 *ab_crt_m1 = new u64[n_expanded]{};
        u64 *ab_crt_m2 = new u64[n_expanded]{};

        // 分别在两个小模数下进行NTT
        poly_multiply_ntt(a, b, ab_crt_m1, n, SPECIAL_CRT_M1);
        poly_multiply_ntt(a, b, ab_crt_m2, n, SPECIAL_CRT_M2);

        for (u64 i = 0; i < n_expanded; ++i) {
            // 合并结果，得到 val mod (SPECIAL_CRT_M1 * SPECIAL_CRT_M2)
            u64 combined_val_mod_M1M2 = crt_combine_special_two(
                ab_crt_m1[i], SPECIAL_CRT_M1,
                ab_crt_m2[i], SPECIAL_CRT_M2
            );
            // 最终结果对目标大模数p取模
            ab[i] = combined_val_mod_M1M2 % p;
        }

        delete[] ab_crt_m1;
        delete[] ab_crt_m2;
    } else {
        // 对于其他模数p，使用原有的三模CRT策略
        u64 **ab_crt = new u64 *[CRT_NUMS];
        u128 *ab_u128 = new u128[n_expanded];
        for (u64 i = 0; i < CRT_NUMS; i++) {
            ab_crt[i] = new u64[n_expanded]{};
            poly_multiply_ntt(a, b, ab_crt[i], n, CRT_MODS[i]); // CRT_MODS是原有的三个模数
        }
        for (u64 i = 0; i < n_expanded; ++i) ab_u128[i] = ab_crt[0][i];
        CRT_combine(ab_u128, ab_crt, n_expanded); // 使用原有的CRT_combine合并三模结果
        for (u64 i = 0; i < n_expanded; ++i) ab[i] = ab_u128[i] % p;
        delete[] ab_u128;
        for (u64 i = 0; i < CRT_NUMS; ++i) delete[] ab_crt[i];
        delete[] ab_crt;
    }
}

void get_rev(u64 *rev, u64 lim) {
    for (u64 i = 0; i < lim; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
    }
}

void ntt(u64 *a, u64 lim, int opt, u64 p) {
    ModGeneric<u64> mod_op(p); // Use ModGeneric for operations modulo p
    u64 *rev = new u64[lim]{};
    get_rev(rev, lim);
    for (u64 i = 0; i < lim; ++i) if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    for (u64 len = 2; len <= lim; len <<= 1) {
        u64 m = len >> 1, wn = mod_op.pow(3, (p - 1) / len); // Use ModGeneric's pow
        if (opt == -1) wn = mod_op.inv(wn); // Use ModGeneric's inv (which uses pow)
        for (u64 i = 0; i < lim; i += len) {
            u64 w = 1;
            for (u64 j = 0; j < m; ++j) {
                u64 u = a[i + j], v = mod_op.mul(a[i + j + m], w); // Use ModGeneric's mul
                a[i + j] = mod_op.add(u, v); // Use ModGeneric's add
                a[i + j + m] = mod_op.sub(u, v); // Use ModGeneric's sub
                w = mod_op.mul(w, wn); // Use ModGeneric's mul
            }
        }
    }
    if (opt == -1) {
        u64 inv = mod_op.inv(lim); // Use ModGeneric's inv
        for (u64 i = 0; i < lim; ++i) a[i] = mod_op.mul(a[i], inv); // Use ModGeneric's mul
    }
    delete[] rev;
}
void poly_multiply_ntt(u64 *a, u64 *b, u64 *ab, u64 n, u64 p_ntt) {
    memset(ab, 0, sizeof(u64) * (2 * n - 1));
    u64 lim = 1; while (lim < 2 * n) lim <<= 1;
    u64 *A = new u64[lim]{}, *B = new u64[lim]{};
    for (u64 i = 0; i < n; ++i) {
        A[i] = a[i] % p_ntt;
        B[i] = b[i] % p_ntt;
    }
    // The rest of A and B (from n to lim-1) are already zero due to new u64[lim]{}
    ntt(A, lim, 1, p_ntt); ntt(B, lim, 1, p_ntt);
    for (u64 i = 0; i < lim; ++i) A[i] = ((u128)A[i] * B[i]) % p_ntt;
    ntt(A, lim, -1, p_ntt);
    for (u64 i = 0; i < 2 * n - 1; ++i) ab[i] = A[i];
    delete[] A; delete[] B;
}
void fRead(u64 *a, u64 *b, u64 *n, u64 *p, int input_id) {
    // 数据输入函数
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin >> *n >> *p;
    for (u64 i = 0; i < *n; i++) {
        int temp;
        fin >> temp;
        a[i] = static_cast<u64>(temp);
    }
    for (u64 i = 0; i < *n; i++) {
        int temp;
        fin >> temp;
        b[i] = static_cast<u64>(temp);
    }
}
void fCheck(u64 *ab, u64 n, int input_id) {
    // 判断多项式乘法结果是否正确
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (u64 i = 0; i < n * 2 - 1; i++) {
        u64 x;
        fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
    return;
}
void fWrite(u64 *ab, u64 n, int input_id) {
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (u64 i = 0; i < n * 2 - 1; i++) {
        fout << ab[i] << '\n';
    }
}
u64 a[300000], b[300000], ab[300000];
int main(int argc, char *argv[]) {
    int test_begin = 0;
    int test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        u64 n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        poly_multiply_crt(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout << "平均延迟 n = " << n_ << " p = " << p_ << " : " << ans << " (ms) " << std::endl;
        fWrite(ab, n_, i);
    }
    return 0;
}
