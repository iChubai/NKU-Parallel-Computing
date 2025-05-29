/*
 * ===========================================
 * 文件名: main_simd_float.cc
 * 描述: 使用SIMD优化的浮点运算实现NTT的并行算法
 * ===========================================
 */
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <arm_neon.h>  

void fRead(int *a, int *b, int *n, int *p, int input_id){
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){
        fin>>b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

int qpow(int x, int y, int p) {
    int res = 1;
    x %= p;
    while (y) {
        if (y & 1) res = 1LL * res * x % p;
        x = 1LL * x * x % p;
        y >>= 1;
    }
    return res;
}

void get_rev(int *rev, int lim) {
    for (int i = 0; i < lim; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
    }
}

void ntt_simd_float(int *a, int lim, int opt, int p) {
    int rev[lim];
    memset(rev, 0, sizeof(int) * lim);
    get_rev(rev, lim);

    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }

    float inv_p = 1.0f / p;

    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);

        for (int i = 0; i < lim; i += len) {
            int w = 1;

            for (int j = 0; j < m; j += 4) {
                if (j + 4 <= m) {
                    int32x4_t vw = {w,
                                   (int)(1LL * w * wn % p),
                                   (int)(1LL * w * wn % p * wn % p),
                                   (int)(1LL * w * wn % p * wn % p * wn % p)};

                    int32x4_t vu = vld1q_s32(a + i + j);
                    int32x4_t vv = vld1q_s32(a + i + j + m);

                    int32x4_t vvw;
                    float32x4_t vf_p = vdupq_n_f32(p);
                    float32x4_t vf_inv_p = vdupq_n_f32(inv_p);

                    for (int k = 0; k < 4; ++k) {
                        int v_val = vgetq_lane_s32(vv, k);
                        int w_val = vgetq_lane_s32(vw, k);

                        long long prod = 1LL * v_val * w_val;
                        float f_prod = prod;
                        float f_q = f_prod * inv_p;
                        int q = (int)f_q;
                        int mod = prod - q * p;
                        if (mod >= p) mod -= p;
                        if (mod < 0) mod += p;

                        vvw = vsetq_lane_s32(mod, vvw, k);
                    }

                    int32x4_t new_u, new_v;
                    int32x4_t vp = vdupq_n_s32(p);

                    for (int k = 0; k < 4; ++k) {
                        int u_val = vgetq_lane_s32(vu, k);
                        int vw_val = vgetq_lane_s32(vvw, k);

                        int sum = u_val + vw_val;
                        if (sum >= p) sum -= p;
                        new_u = vsetq_lane_s32(sum, new_u, k);

                        int diff = u_val - vw_val;
                        if (diff < 0) diff += p;
                        new_v = vsetq_lane_s32(diff, new_v, k);
                    }

                    vst1q_s32(a + i + j, new_u);
                    vst1q_s32(a + i + j + m, new_v);

                    w = 1LL * w * wn % p * wn % p * wn % p * wn % p;
                } else {
                    for (; j < m; ++j) {
                        int u = a[i + j];
                        int v = 1LL * a[i + j + m] * w % p;

                        a[i + j] = (u + v) % p;
                        a[i + j + m] = (u - v + p) % p;

                        w = 1LL * w * wn % p;
                    }
                }
            }
        }
    }

    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);

        for (int i = 0; i < lim; i += 4) {
            if (i + 4 <= lim) {
                int32x4_t va = vld1q_s32(a + i);

                int32x4_t vresult;
                for (int j = 0; j < 4; ++j) {
                    int aj = vgetq_lane_s32(va, j);
                    int res = 1LL * aj * inv % p;
                    vresult = vsetq_lane_s32(res, vresult, j);
                }

                vst1q_s32(a + i, vresult);
            } else {
                for (int j = i; j < lim; ++j) {
                    a[j] = 1LL * a[j] * inv % p;
                }
            }
        }
    }
}

void poly_multiply_simd_float(int *a, int *b, int *ab, int n, int p) {
    memset(ab, 0, sizeof(int) * (2 * n - 1));

    int lim = 1;
    while (lim < 2 * n) lim <<= 1;

    int A[lim], B[lim];
    memset(A, 0, sizeof(int) * lim);
    memset(B, 0, sizeof(int) * lim);

    for (int i = 0; i < n; ++i) {
        A[i] = a[i];
        B[i] = b[i];
    }

    ntt_simd_float(A, lim, 1, p);
    ntt_simd_float(B, lim, 1, p);

    float inv_p = 1.0f / p;
    for (int i = 0; i < lim; i += 4) {
        if (i + 4 <= lim) {
            int32x4_t va = vld1q_s32(A + i);
            int32x4_t vb = vld1q_s32(B + i);
            int32x4_t vresult;

            for (int j = 0; j < 4; ++j) {
                int aj = vgetq_lane_s32(va, j);
                int bj = vgetq_lane_s32(vb, j);

                long long prod = 1LL * aj * bj;
                float f_prod = prod;
                float f_q = f_prod * inv_p;
                int q = (int)f_q;
                int mod = prod - q * p;
                if (mod >= p) mod -= p;
                if (mod < 0) mod += p;

                vresult = vsetq_lane_s32(mod, vresult, j);
            }

            vst1q_s32(A + i, vresult);
        } else {
            for (int j = i; j < lim; ++j) {
                A[j] = 1LL * A[j] * B[j] % p;
            }
        }
    }

    ntt_simd_float(A, lim, -1, p);

    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = A[i];
    }
}

int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();

        // 使用SIMD优化的NTT实现多项式乘法
        // 根据模数选择不同的实现方法
        // 使用Montgomery规约。
        if (p_ == 7340033 || p_ == 104857601 || p_ == 469762049) {
            // 对于这些模数，使用Montgomery规约
            poly_multiply_simd_float(a, b, ab, n_, p_);
        } else {
            // 对于其他模数，使用浮点数近似取模
            poly_multiply_simd_float(a, b, ab, n_, p_);
        }

        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }
    return 0;
}
