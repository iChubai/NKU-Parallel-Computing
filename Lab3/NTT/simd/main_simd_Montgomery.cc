/*
 * ===========================================
 * 文件名: main_simd_Montgomery.cc
 * 描述: 使用Montgomery规约实现NTT的并行算法
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

// Montgomery规约相关常量和函数
typedef uint32_t u32;
typedef uint64_t u64;

// 辅助函数：计算模逆元
u32 inverse(u32 a, u32 m) {
    u32 u = 0, v = 1;
    u32 m0 = m;
    while (a != 0) {
        u32 q = m / a;
        u32 r = m % a;
        u32 t = u - q * v;
        m = a;
        a = r;
        u = v;
        v = t;
    }
    if (u < 0) u += m0;
    return u;
}   

// Montgomery规约的向量化实现
class MontgomeryReducer {
private:
    u32 mod;       
    u32 mod_inv;   
    u32 r2;        

public:
    MontgomeryReducer(u32 mod) : mod(mod) {
        mod_inv = inverse(mod, 1U << 31);  
        r2 = 0;
        u64 r = 1;
        for (int i = 0; i < 64; ++i) {
            r = (r << 1) % mod;
        }
        r2 = r;
    }

    // 单个数值的Montgomery规约
    u32 reduce(u64 x) {
        u32 q = (u32)x * mod_inv;  // q = x * mod' mod 2^32
        u64 m = (u64)q * mod;      // m = q * mod
        u32 y = (x - m) >> 32;     // y = (x - m) / 2^32
        return x < m ? y + mod : y; // 保证结果非负
    }

    // 将数值转换到Montgomery域
    u32 to_montgomery(u32 x) {
        return reduce((u64)x * r2);
    }

    // 将数值从Montgomery域转换回普通域
    u32 from_montgomery(u32 x) {
        return reduce((u64)x);
    }

    // 在Montgomery域中进行乘法
    u32 mul(u32 a, u32 b) {
        return reduce((u64)a * b);
    }

    // 向量化的Montgomery乘法
    void mul_vector(u32* a, u32* b, u32* result, int n) {
        for (int i = 0; i < n; i += 4) {
            // 加载4个元素到NEON寄存器
            uint32x4_t va = vld1q_u32(a + i);
            uint32x4_t vb = vld1q_u32(b + i);

            // 存储结果
            uint32x4_t vresult;

            // 逐个处理每个元素（因为NEON不直接支持64位乘法）
            for (int j = 0; j < 4; ++j) {
                u32 ai = vgetq_lane_u32(va, j);
                u32 bi = vgetq_lane_u32(vb, j);
                u32 res = mul(ai, bi);
                vresult = vsetq_lane_u32(res, vresult, j);
            }

            // 存储结果
            vst1q_u32(result + i, vresult);
        }
    }
};

void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
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
    // 判断多项式乘法结果是否正确
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
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
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

// 快速幂函数，用于计算 x^y mod p
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

// 计算位反转表
void get_rev(int *rev, int lim) {
    for (int i = 0; i < lim; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
    }
}

// SIMD优化的NTT变换
void ntt_simd(int *a, int lim, int opt, int p) {
    // 创建Montgomery规约器
    MontgomeryReducer reducer(p);

    // 位反转置换
    int rev[lim];
    memset(rev, 0, sizeof(int) * lim);
    get_rev(rev, lim);

    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }

    // 将所有数转换到Montgomery域
    for (int i = 0; i < lim; ++i) {
        a[i] = reducer.to_montgomery(a[i]);
    }

    // 蝴蝶操作
    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
        // 原根，对于不同的模数p，可能需要不同的原根
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);

        // 将wn转换到Montgomery域
        wn = reducer.to_montgomery(wn);
    
        // 对于长度足够的段，使用SIMD优化
        for (int i = 0; i < lim; i += len) {
            int w = reducer.to_montgomery(1);  // 在Montgomery域中的1

            // 处理每个蝴蝶操作
            for (int j = 0; j < m; j += 4) {
                if (j + 4 <= m) {
                    // 加载4个w值
                    uint32x4_t vw = {(u32)w,
                                    reducer.mul(w, wn),
                                    reducer.mul(reducer.mul(w, wn), wn),
                                    reducer.mul(reducer.mul(reducer.mul(w, wn), wn), wn)};

                    // 加载4对数据
                    uint32x4_t vu = vld1q_u32((u32*)(a + i + j));
                    uint32x4_t vv = vld1q_u32((u32*)(a + i + j + m));

                    // 计算v * w (在Montgomery域中)
                    uint32x4_t vvw;
                    for (int k = 0; k < 4; ++k) {
                        u32 vk = vgetq_lane_u32(vv, k);
                        u32 wk = vgetq_lane_u32(vw, k);
                        vvw = vsetq_lane_u32(reducer.mul(vk, wk), vvw, k);
                    }

                    // 计算新的u和v
                    uint32x4_t new_u, new_v;
                    for (int k = 0; k < 4; ++k) {
                        u32 uk = vgetq_lane_u32(vu, k);
                        u32 vwk = vgetq_lane_u32(vvw, k);

                        // u + v*w
                        u32 sum = uk + vwk;
                        if (sum >= p) sum -= p;
                        new_u = vsetq_lane_u32(sum, new_u, k);

                        // u - v*w + p
                        u32 diff = uk >= vwk ? uk - vwk : uk + p - vwk;
                        new_v = vsetq_lane_u32(diff, new_v, k);
                    }

                    // 存储结果
                    vst1q_u32((u32*)(a + i + j), new_u);
                    vst1q_u32((u32*)(a + i + j + m), new_v);

                    // 更新w
                    w = reducer.mul(reducer.mul(reducer.mul(reducer.mul(w, wn), wn), wn), wn);
                } else {
                    // 处理剩余的元素（不足4个）
                    for (; j < m; ++j) {
                        int u = a[i + j];
                        int v = reducer.mul(a[i + j + m], w);

                        a[i + j] = u + v >= p ? u + v - p : u + v;
                        a[i + j + m] = u >= v ? u - v : u + p - v;

                        w = reducer.mul(w, wn);
                    }
                }
            }
        }
    }

    // 如果是逆变换，需要除以lim
    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);
        inv = reducer.to_montgomery(inv);

        // 向量化乘法
        for (int i = 0; i < lim; i += 4) {
            if (i + 4 <= lim) {
                uint32x4_t va = vld1q_u32((u32*)(a + i));
                uint32x4_t vinv = vdupq_n_u32(inv);  // 广播inv到所有元素

                // 计算a * inv
                uint32x4_t vresult;
                for (int j = 0; j < 4; ++j) {
                    u32 aj = vgetq_lane_u32(va, j);
                    vresult = vsetq_lane_u32(reducer.mul(aj, inv), vresult, j);
                }

                vst1q_u32((u32*)(a + i), vresult);
            } else {
                // 处理剩余元素
                for (int j = i; j < lim; ++j) {
                    a[j] = reducer.mul(a[j], inv);
                }
            }
        }
    }

    // 将结果从Montgomery域转换回普通域
    for (int i = 0; i < lim; ++i) {
        a[i] = reducer.from_montgomery(a[i]);
    }
}

// 多项式乘法函数（使用SIMD优化的NTT）
void poly_multiply_simd(int *a, int *b, int *ab, int n, int p) {
    // 清空结果数组
    memset(ab, 0, sizeof(int) * (2 * n - 1));

    // 计算NTT需要的长度（2的幂次）
    int lim = 1;
    while (lim < 2 * n) lim <<= 1;

    // 创建临时数组
    int A[lim], B[lim];
    memset(A, 0, sizeof(int) * lim);
    memset(B, 0, sizeof(int) * lim);

    // 复制输入数组到临时数组
    for (int i = 0; i < n; ++i) {
        A[i] = a[i];
        B[i] = b[i];
    }

    // 执行SIMD优化的NTT变换
    ntt_simd(A, lim, 1, p);
    ntt_simd(B, lim, 1, p);

    // 点乘（使用SIMD优化）
    for (int i = 0; i < lim; i += 4) {
        if (i + 4 <= lim) {
            int32x4_t va = vld1q_s32(A + i);
            int32x4_t vb = vld1q_s32(B + i);
            int32x4_t vresult;
            int32x4_t vp = vdupq_n_s32(p);

            // 计算A[i] * B[i] % p
            for (int j = 0; j < 4; ++j) {
                int aj = vgetq_lane_s32(va, j);
                int bj = vgetq_lane_s32(vb, j);
                int res = 1LL * aj * bj % p;
                vresult = vsetq_lane_s32(res, vresult, j);
            }

            vst1q_s32(A + i, vresult);
        } else {
            // 处理剩余元素
            for (int j = i; j < lim; ++j) {
                A[j] = 1LL * A[j] * B[j] % p;
            }
        }
    }

    // 执行逆NTT变换
    ntt_simd(A, lim, -1, p);

    // 复制结果到输出数组
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
            poly_multiply_simd(a, b, ab, n_, p_);
        } else {
            poly_multiply_simd(a, b, ab, n_, p_);
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
