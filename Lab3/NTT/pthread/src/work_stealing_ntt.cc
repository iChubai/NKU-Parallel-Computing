/*
 * =============================================================================
 * 文件名: work_stealing_ntt.cc
 * 描述: Work-Stealing NTT并行调度器
 * 核心原理: 动态负载均衡，每个线程维护任务队列，空闲时从其他线程"偷取"任务
 * 优势: 解决传统静态分配中的负载不均衡问题，提高CPU利用率
 * =============================================================================
 */

#include <atomic>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <random>
#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <cstring>

using u32 = uint32_t;
using u64 = uint64_t;

/**
 * 无锁环形缓冲区
 * 用于Work-Stealing队列的高效实现
 */
template<typename T, size_t N>
class LockFreeRingBuffer {
private:
    static_assert((N & (N - 1)) == 0, "Size must be power of 2");
    static constexpr size_t MASK = N - 1;
    
    alignas(64) std::atomic<size_t> head{0};  // 生产者索引
    alignas(64) std::atomic<size_t> tail{0};  // 消费者索引
    alignas(64) T buffer[N];

public:
    /**
     * 尝试推入一个元素 (从头部)
     * @param item: 要推入的元素
     * @return: 成功返回true，失败返回false
     */
    bool try_push(T&& item) {
        size_t current_head = head.load(std::memory_order_relaxed);
        size_t next_head = (current_head + 1) & MASK;
        
        if (next_head == tail.load(std::memory_order_acquire)) {
            return false;  // 队列满
        }
        
        buffer[current_head] = std::move(item);
        head.store(next_head, std::memory_order_release);
        return true;
    }

    /**
     * 尝试弹出一个元素 (从头部)
     * @param item: 存储弹出元素的引用
     * @return: 成功返回true，失败返回false
     */
    bool try_pop(T& item) {
        size_t current_head = head.load(std::memory_order_relaxed);
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false;  // 队列空
        }
        
        size_t prev_head = (current_head - 1) & MASK;
        item = std::move(buffer[prev_head]);
        head.store(prev_head, std::memory_order_release);
        return true;
    }

    /**
     * 尝试偷取一个元素 (从尾部)
     * @param item: 存储偷取元素的引用
     * @return: 成功返回true，失败返回false
     */
    bool try_steal(T& item) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        if (current_tail == head.load(std::memory_order_acquire)) {
            return false;  // 队列空
        }
        
        item = std::move(buffer[current_tail]);
        tail.store((current_tail + 1) & MASK, std::memory_order_release);
        return true;
    }

    /**
     * 检查队列是否为空
     */
    bool empty() const {
        return head.load(std::memory_order_relaxed) == tail.load(std::memory_order_relaxed);
    }

    /**
     * 获取队列大小 (近似值)
     */
    size_t size() const {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_relaxed);
        return (h - t) & MASK;
    }
};

/**
 * NTT任务定义
 * 包含执行一个NTT子任务所需的所有信息
 */
struct NTTTask {
    enum TaskType {
        BUTTERFLY_STAGE,    // 蝶形运算阶段
        BITREV_STAGE,       // 位反转阶段
        POINTWISE_MUL,      // 点乘阶段
        INVERSE_SCALE       // 逆变换缩放阶段
    };

    TaskType type;
    u64* data;              // 数据指针
    size_t start_idx;       // 起始索引
    size_t end_idx;         // 结束索引
    size_t stride;          // 内存步长
    size_t level;           // NTT层级
    u64 twiddle_factor;     // 旋转因子
    u64 modulus;            // 模数
    bool inverse;           // 是否为逆变换
    
    // 依赖任务管理
    std::atomic<int> ref_count{1};      // 引用计数
    std::vector<std::shared_ptr<NTTTask>> dependencies;  // 依赖的任务
    std::vector<std::weak_ptr<NTTTask>> dependents;      // 依赖此任务的任务

    NTTTask(TaskType t, u64* d, size_t start, size_t end, u64 mod) 
        : type(t), data(d), start_idx(start), end_idx(end), modulus(mod) {}
};

/**
 * Work-Stealing NTT调度器
 * 核心组件：每个工作线程维护自己的任务队列
 */
class WorkStealingNTTScheduler {
private:
    static constexpr size_t QUEUE_SIZE = 1024;
    static constexpr size_t MAX_STEAL_ATTEMPTS = 8;
    static constexpr int BACKOFF_BASE_DELAY = 10;  // 微秒
    static constexpr size_t MAX_TASKS = 100000;     // 防止任务爆炸

    struct WorkerThread {
        std::thread thread;
        LockFreeRingBuffer<std::shared_ptr<NTTTask>, QUEUE_SIZE> local_queue;
        std::atomic<bool> working{false};
        std::atomic<bool> should_exit{false};  // 改进退出机制
        std::atomic<size_t> tasks_completed{0};
        std::atomic<size_t> tasks_stolen{0};
        std::atomic<size_t> steal_attempts{0};
        
        // 线程本地随机数生成器 (避免争用)
        thread_local static std::mt19937 rng;
    };

    std::vector<std::unique_ptr<WorkerThread>> workers;
    std::atomic<bool> shutdown{false};
    std::atomic<size_t> active_tasks{0};
    std::atomic<size_t> total_tasks_created{0};  // 防止任务过度创建
    std::condition_variable completion_cv;
    std::mutex completion_mutex;

    // 统计信息
    std::atomic<size_t> total_tasks_processed{0};
    std::atomic<size_t> total_steal_operations{0};
    std::atomic<size_t> total_failed_steals{0};

public:
    /**
     * 构造函数
     * @param num_threads: 工作线程数量
     */
    explicit WorkStealingNTTScheduler(int num_threads = std::thread::hardware_concurrency()) {
        workers.reserve(num_threads);
        
        for (int i = 0; i < num_threads; ++i) {
            auto worker = std::make_unique<WorkerThread>();
            worker->thread = std::thread(&WorkStealingNTTScheduler::worker_main_loop, this, i);
            workers.push_back(std::move(worker));
        }
    }

    /**
     * 析构函数 - 改进的优雅关闭
     */
    ~WorkStealingNTTScheduler() {
        shutdown.store(true);
        
        // 通知所有工作线程退出
        for (auto& worker : workers) {
            worker->should_exit.store(true);
        }
        
        // 等待所有线程完成
        for (auto& worker : workers) {
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }
    }

    /**
     * 执行Work-Stealing NTT - 带安全检查
     */
    void parallel_ntt_work_stealing(u64* data, size_t n, u64 modulus, bool inverse = false) {
        if (!data || n == 0 || (n & (n - 1)) != 0) {
            throw std::invalid_argument("Invalid input: data is null or n is not power of 2");
        }
        
        if (n <= 64) {
            // 小规模问题直接串行处理
            traditional_ntt(data, n, modulus, inverse);
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // 重置统计信息
        total_tasks_processed.store(0);
        total_steal_operations.store(0);
        total_failed_steals.store(0);
        total_tasks_created.store(0);

        try {
            // 分解任务并分发
            auto task_graph = decompose_ntt_to_tasks_safe(data, n, modulus, inverse);
            distribute_initial_tasks_safe(task_graph);

            // 等待所有任务完成
            wait_for_completion_safe();
        } catch (const std::exception& e) {
            std::cerr << "Work-Stealing NTT error: " << e.what() << std::endl;
            // 清理并回退到传统实现
            traditional_ntt(data, n, modulus, inverse);
            return;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // 打印性能统计
        print_performance_stats(duration.count(), n);
    }

private:
    /**
     * 工作线程主循环 - 改进的错误处理
     */
    void worker_main_loop(int worker_id) {
        auto& worker = *workers[worker_id];
        std::shared_ptr<NTTTask> task;
        
        // 为每个线程设置随机种子
        worker.rng.seed(std::chrono::steady_clock::now().time_since_epoch().count() + worker_id);

        try {
            while (!shutdown.load(std::memory_order_relaxed) && !worker.should_exit.load()) {
                bool found_work = false;

                // 1. 尝试从本地队列获取任务
                if (worker.local_queue.try_pop(task)) {
                    found_work = true;
                } else {
                    // 2. 尝试从其他线程偷取任务
                    found_work = try_steal_task_safe(worker_id, task);
                }

                if (found_work && task) {
                    worker.working.store(true);
                    execute_task_safe(task, worker_id);
                    worker.tasks_completed.fetch_add(1);
                    worker.working.store(false);
                    
                    // 任务完成后可能激活新任务
                    activate_dependent_tasks_safe(task);
                    
                    // 清理任务引用
                    task.reset();
                } else {
                    // 没有找到工作，短暂休眠
                    adaptive_backoff(worker_id);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Worker " << worker_id << " error: " << e.what() << std::endl;
        }
    }

    /**
     * 安全的任务窃取
     */
    bool try_steal_task_safe(int worker_id, std::shared_ptr<NTTTask>& task) {
        auto& worker = *workers[worker_id];
        worker.steal_attempts.fetch_add(1);

        // 随机选择一个受害者线程
        std::uniform_int_distribution<int> dist(0, workers.size() - 1);
        
        for (size_t attempt = 0; attempt < MAX_STEAL_ATTEMPTS; ++attempt) {
            int victim_id = dist(worker.rng);
            
            if (victim_id == worker_id) continue;  // 不从自己偷取

            auto& victim = *workers[victim_id];
            
            // 只从工作繁忙的线程偷取
            if (victim.working.load() && victim.local_queue.try_steal(task)) {
                if (task) {  // 确保任务有效
                    worker.tasks_stolen.fetch_add(1);
                    total_steal_operations.fetch_add(1);
                    return true;
                }
            }
        }

        total_failed_steals.fetch_add(1);
        return false;
    }

    /**
     * 安全的任务执行
     */
    void execute_task_safe(const std::shared_ptr<NTTTask>& task, int /*worker_id*/) {
        if (!task || !task->data) {
            return;  // 跳过无效任务
        }
        
        try {
            switch (task->type) {
                case NTTTask::BUTTERFLY_STAGE:
                    execute_butterfly_stage_safe(task);
                    break;
                case NTTTask::BITREV_STAGE:
                    execute_bitrev_stage_safe(task);
                    break;
                case NTTTask::POINTWISE_MUL:
                    execute_pointwise_mul_safe(task);
                    break;
                case NTTTask::INVERSE_SCALE:
                    execute_inverse_scale_safe(task);
                    break;
            }
            
            total_tasks_processed.fetch_add(1);
        } catch (const std::exception& e) {
            std::cerr << "Task execution error: " << e.what() << std::endl;
        }
    }

    /**
     * 安全的蝶形运算执行
     */
    void execute_butterfly_stage_safe(const std::shared_ptr<NTTTask>& task) {
        if (task->end_idx <= task->start_idx || task->stride == 0) {
            return;  // 无效范围
        }
        
        FastModular mod_op(task->modulus);
        
        for (size_t i = task->start_idx; i < task->end_idx && i + task->stride < task->end_idx; 
             i += task->stride * 2) {
            size_t j = i + task->stride;
            
            // 边界检查
            if (j >= task->end_idx) break;
            
            u64 u = task->data[i];
            u64 v = mod_op.mul(task->data[j], task->twiddle_factor);
            
            task->data[i] = mod_op.add(u, v);
            task->data[j] = mod_op.sub(u, v);
        }
    }

    /**
     * 安全的位反转执行
     */
    void execute_bitrev_stage_safe(const std::shared_ptr<NTTTask>& task) {
        if (task->end_idx <= task->start_idx) {
            return;
        }
        
        size_t n = task->end_idx - task->start_idx;
        u64* data = task->data + task->start_idx;
        
        for (size_t i = 1, j = 0; i < n; ++i) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j && i < n && j < n) {  // 双重边界检查
                std::swap(data[i], data[j]);
            }
        }
    }

    /**
     * 安全的点乘执行
     */
    void execute_pointwise_mul_safe(const std::shared_ptr<NTTTask>& task) {
        if (task->end_idx <= task->start_idx) {
            return;
        }
        
        FastModular mod_op(task->modulus);
        for (size_t i = task->start_idx; i < task->end_idx; ++i) {
            task->data[i] = mod_op.mul(task->data[i], task->data[i]);  // 示例
        }
    }

    /**
     * 安全的逆变换缩放执行
     */
    void execute_inverse_scale_safe(const std::shared_ptr<NTTTask>& task) {
        if (task->end_idx <= task->start_idx) {
            return;
        }
        
        FastModular mod_op(task->modulus);
        u64 inv_factor = task->twiddle_factor;
        
        for (size_t i = task->start_idx; i < task->end_idx; ++i) {
            task->data[i] = mod_op.mul(task->data[i], inv_factor);
        }
    }

    /**
     * 安全的依赖任务激活
     */
    void activate_dependent_tasks_safe(const std::shared_ptr<NTTTask>& completed_task) {
        if (!completed_task) return;
        
        // 创建依赖任务的副本避免迭代器失效
        std::vector<std::weak_ptr<NTTTask>> dependents_copy;
        dependents_copy.reserve(completed_task->dependents.size());
        
        for (auto& weak_dependent : completed_task->dependents) {
            dependents_copy.push_back(weak_dependent);
        }
        
        for (auto& weak_dependent : dependents_copy) {
            if (auto dependent = weak_dependent.lock()) {
                int old_ref = dependent->ref_count.fetch_sub(1);
                if (old_ref == 1) {
                    // 所有依赖都完成，可以执行此任务
                    schedule_task_safe(dependent);
                }
            }
        }
    }

    /**
     * 安全的任务调度
     */
    void schedule_task_safe(std::shared_ptr<NTTTask> task) {
        if (!task) return;
        
        // 寻找负载最轻的线程
        int best_worker = find_least_loaded_worker();
        
        if (!workers[best_worker]->local_queue.try_push(std::move(task))) {
            // 队列满，尝试其他线程
            for (size_t i = 0; i < workers.size(); ++i) {
                int worker_id = (best_worker + i) % workers.size();
                if (workers[worker_id]->local_queue.try_push(std::move(task))) {
                    active_tasks.fetch_add(1);
                    return;
                }
            }
            // 所有队列都满，直接执行 (降级策略)
            if (task) {
                execute_task_safe(task, 0);
            }
        } else {
            active_tasks.fetch_add(1);
        }
    }

    /**
     * 安全的任务分解
     */
    std::vector<std::shared_ptr<NTTTask>> decompose_ntt_to_tasks_safe(u64* data, size_t n, 
                                                                     u64 modulus, bool inverse) {
        std::vector<std::shared_ptr<NTTTask>> tasks;
        
        // 防止任务爆炸
        if (total_tasks_created.load() > MAX_TASKS) {
            throw std::runtime_error("Too many tasks created, falling back to traditional NTT");
        }
        
        // 简化的任务分解，避免复杂的依赖关系
        
        // 1. 位反转任务
        auto bitrev_task = std::make_shared<NTTTask>(NTTTask::BITREV_STAGE, data, 0, n, modulus);
        bitrev_task->inverse = inverse;
        tasks.push_back(bitrev_task);
        total_tasks_created.fetch_add(1);

        // 2. 简化的蝶形运算任务 (减少层数避免过度分解)
        size_t max_parallel_levels = std::min(static_cast<size_t>(8), 
                                             static_cast<size_t>(std::log2(n)));
        
        for (size_t level = 0; level < max_parallel_levels && total_tasks_created.load() < MAX_TASKS; 
             ++level) {
            size_t len = 1ULL << (level + 1);
            if (len > n) break;
            
            size_t stride = len / 2;
            size_t num_blocks = n / len;
            
            FastModular mod_op(modulus);
            u64 wlen = mod_op.pow(3, (modulus - 1) / len);
            if (inverse) wlen = mod_op.pow(wlen, modulus - 2);
            
            for (size_t block = 0; block < num_blocks && total_tasks_created.load() < MAX_TASKS; 
                 ++block) {
                size_t block_start = block * len;
                size_t block_end = block_start + stride;
                
                auto task = std::make_shared<NTTTask>(NTTTask::BUTTERFLY_STAGE, 
                                                    data, block_start, block_end, modulus);
                task->stride = stride;
                task->level = len;
                task->twiddle_factor = wlen;
                task->inverse = inverse;
                
                // 简化依赖关系 - 只依赖前一个任务
                if (!tasks.empty()) {
                    task->ref_count = 1;
                    task->dependencies.push_back(tasks.back());
                    tasks.back()->dependents.push_back(task);
                }
                
                tasks.push_back(task);
                total_tasks_created.fetch_add(1);
            }
        }

        return tasks;
    }

    /**
     * 安全的初始任务分发
     */
    void distribute_initial_tasks_safe(const std::vector<std::shared_ptr<NTTTask>>& tasks) {
        for (auto& task : tasks) {
            if (task && task->dependencies.empty()) {
                schedule_task_safe(task);
            }
        }
    }

    /**
     * 安全的完成等待
     */
    void wait_for_completion_safe() {
        auto start_wait = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(30);  // 30秒超时
        
        std::unique_lock<std::mutex> lock(completion_mutex);
        bool completed = completion_cv.wait_for(lock, timeout, [this]() {
            // 检查是否所有线程都空闲且队列为空
            bool all_idle = true;
            for (auto& worker : workers) {
                if (worker->working.load() || !worker->local_queue.empty()) {
                    all_idle = false;
                    break;
                }
            }
            return all_idle && active_tasks.load() == 0;
        });
        
        if (!completed) {
            throw std::runtime_error("Work-stealing NTT completion timeout");
        }
    }

    /**
     * 自适应退避策略
     */
    void adaptive_backoff(int /*worker_id*/) {
        static thread_local int backoff_count = 0;
        
        if (backoff_count < 10) {
            // 短暂yield
            std::this_thread::yield();
        } else if (backoff_count < 100) {
            // 短暂睡眠
            std::this_thread::sleep_for(std::chrono::microseconds(BACKOFF_BASE_DELAY));
        } else {
            // 长时间睡眠
            std::this_thread::sleep_for(std::chrono::microseconds(BACKOFF_BASE_DELAY * 10));
        }
        
        backoff_count = (backoff_count + 1) % 1000;
    }

    /**
     * 打印性能统计信息
     */
    void print_performance_stats(long duration_us, size_t n) {
        std::cout << "\n=== Work-Stealing NTT性能统计 ===" << std::endl;
        std::cout << "数据长度: " << n << std::endl;
        std::cout << "执行时间: " << duration_us << " 微秒" << std::endl;
        std::cout << "总任务数: " << total_tasks_processed.load() << std::endl;
        std::cout << "偷取操作数: " << total_steal_operations.load() << std::endl;
        std::cout << "失败偷取数: " << total_failed_steals.load() << std::endl;
        
        double steal_success_rate = total_steal_operations.load() > 0 ?
            100.0 * total_steal_operations.load() / 
            (total_steal_operations.load() + total_failed_steals.load()) : 0.0;
        std::cout << "偷取成功率: " << steal_success_rate << "%" << std::endl;
        
        std::cout << "\n各线程统计:" << std::endl;
        for (size_t i = 0; i < workers.size(); ++i) {
            auto& worker = *workers[i];
            std::cout << "线程 " << i << ": 完成任务 " << worker.tasks_completed.load()
                     << ", 偷取任务 " << worker.tasks_stolen.load()
                     << ", 偷取尝试 " << worker.steal_attempts.load() << std::endl;
        }
    }

    /**
     * 传统NTT实现 (用于小规模问题)
     */
    void traditional_ntt(u64* data, size_t n, u64 modulus, bool inverse) {
        // 简化的传统NTT实现
        FastModular mod_op(modulus);
        
        // 位反转
        for (size_t i = 1, j = 0; i < n; ++i) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(data[i], data[j]);
        }

        // 蝶形运算
        for (size_t len = 2; len <= n; len <<= 1) {
            u64 wlen = mod_op.pow(3, (modulus - 1) / len);
            if (inverse) wlen = mod_op.pow(wlen, modulus - 2);

            for (size_t i = 0; i < n; i += len) {
                u64 w = 1;
                for (size_t j = 0; j < len / 2; ++j) {
                    u64 u = data[i + j];
                    u64 v = mod_op.mul(data[i + j + len / 2], w);
                    data[i + j] = mod_op.add(u, v);
                    data[i + j + len / 2] = mod_op.sub(u, v);
                    w = mod_op.mul(w, wlen);
                }
            }
        }

        // 逆变换缩放
        if (inverse) {
            u64 inv_n = mod_op.pow(n, modulus - 2);
            for (size_t i = 0; i < n; ++i) {
                data[i] = mod_op.mul(data[i], inv_n);
            }
        }
    }

    /**
     * 简化的快速模运算类
     */
    class FastModular {
    private:
        u64 mod;
    public:
        explicit FastModular(u64 m) : mod(m) {}
        
        u64 add(u64 a, u64 b) const {
            u64 result = a + b;
            return result >= mod ? result - mod : result;
        }
        
        u64 sub(u64 a, u64 b) const {
            return a >= b ? a - b : a + mod - b;
        }
        
        u64 mul(u64 a, u64 b) const {
            return ((__uint128_t)a * b) % mod;
        }
        
        u64 pow(u64 base, u64 exp) const {
            u64 result = 1;
            base %= mod;
            while (exp > 0) {
                if (exp & 1) result = mul(result, base);
                base = mul(base, base);
                exp >>= 1;
            }
            return result;
        }
    };

    /**
     * 寻找负载最轻的工作线程
     */
    int find_least_loaded_worker() {
        int best = 0;
        size_t min_load = workers[0]->local_queue.size();
        
        for (size_t i = 1; i < workers.size(); ++i) {
            size_t load = workers[i]->local_queue.size();
            if (load < min_load) {
                min_load = load;
                best = i;
            }
        }
        
        return best;
    }
};

// 定义thread_local静态成员
thread_local std::mt19937 WorkStealingNTTScheduler::WorkerThread::rng;

/**
 * Work-Stealing多项式乘法接口
 */
void work_stealing_poly_multiply(const u64* a, const u64* b, u64* result, 
                               size_t n, u64 modulus, int num_threads = 4) {
    // 扩展到2的幂
    size_t ntt_size = 1;
    while (ntt_size < 2 * n) ntt_size <<= 1;

    WorkStealingNTTScheduler scheduler(num_threads);
    
    std::vector<u64> poly_a(ntt_size, 0), poly_b(ntt_size, 0);
    
    // 复制输入数据
    std::copy(a, a + n, poly_a.begin());
    std::copy(b, b + n, poly_b.begin());

    // 正向NTT
    scheduler.parallel_ntt_work_stealing(poly_a.data(), ntt_size, modulus, false);
    scheduler.parallel_ntt_work_stealing(poly_b.data(), ntt_size, modulus, false);

    // 点乘
    for (size_t i = 0; i < ntt_size; ++i) {
        poly_a[i] = ((__uint128_t)poly_a[i] * poly_b[i]) % modulus;
    }

    // 逆向NTT
    scheduler.parallel_ntt_work_stealing(poly_a.data(), ntt_size, modulus, true);

    // 复制结果
    std::copy(poly_a.begin(), poly_a.begin() + 2 * n - 1, result);
}

// 演示函数
void demonstrate_work_stealing_ntt() {
    std::cout << "=== Work-Stealing NTT算法演示 ===" << std::endl;
    
    const size_t n = 2048;
    const u64 modulus = 998244353;  // NTT友好的质数
    
    std::vector<u64> a(n), b(n), result(2 * n - 1);
    
    // 初始化测试数据
    for (size_t i = 0; i < n; ++i) {
        a[i] = i + 1;
        b[i] = (i * 3 + 7) % modulus;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 执行Work-Stealing多项式乘法
    work_stealing_poly_multiply(a.data(), b.data(), result.data(), n, modulus, 8);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n总体性能:" << std::endl;
    std::cout << "多项式长度: " << n << std::endl;
    std::cout << "总执行时间: " << duration.count() << " 微秒" << std::endl;
    std::cout << "前几项结果: ";
    for (size_t i = 0; i < 10 && i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
}

#ifdef WORK_STEALING_MAIN
int main() {
    demonstrate_work_stealing_ntt();
    return 0;
}
#endif 