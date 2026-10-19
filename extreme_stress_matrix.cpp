#include <dlfcn.h>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstdlib>

// Metrics trackers
std::atomic<long> ops_math{0};
std::atomic<long> ops_queue{0};
std::atomic<long> ops_arena{0};
std::atomic<long> system_faults{0};
std::atomic<bool> test_active{true};

// Function pointer signatures mapped to libTessSDK.so exported symbols[span_2](start_span)[span_2](end_span)
typedef void* (*tess_create_t)();
typedef void  (*tess_destroy_t)(void*);
typedef int   (*tess_process_frame_t)(void*);
typedef void* (*tess_mpmc_create_t)(size_t);
typedef int   (*tess_mpmc_push_t)(void*, void*);
typedef void* (*tess_mpmc_pop_t)(void*);
typedef void  (*tess_mpmc_destroy_t)(void*);
typedef void* (*arena_init_t)(size_t);
typedef void* (*arena_alloc_t)(void*, size_t);
typedef void  (*arena_reset_t)(void*);
typedef void  (*arena_free_t)(void*);

// Subsystem 1: Frame Processing & Vector Pipeline Worker
void frame_pipeline_worker(void* handle) {
    auto create = (tess_create_t)dlsym(handle, "tess_create");
    auto destroy = (tess_destroy_t)dlsym(handle, "tess_destroy");
    auto process = (tess_process_frame_t)dlsym(handle, "tess_process_frame");
    if (!create || !process) return;

    void* ctx = create();
    while (test_active.load(std::memory_order_relaxed)) {
        int res = process(ctx);
        ops_math.fetch_add(1, std::memory_order_relaxed);
        if (res != 0) system_faults.fetch_add(1, std::memory_order_relaxed);
    }
    if (destroy && ctx) destroy(ctx);
}

// Subsystem 2: Lock-Free MPMC Queue Contention Worker
void queue_contention_worker(void* handle, void* queue) {
    auto push = (tess_mpmc_push_t)dlsym(handle, "tess_mpmc_queue_push");
    auto pop = (tess_mpmc_pop_t)dlsym(handle, "tess_mpmc_queue_pop");
    if (!push || !pop) return;

    void* payload = (void*)0xDEADBEEF;
    while (test_active.load(std::memory_order_relaxed)) {
        push(queue, payload);
        pop(queue);
        ops_queue.fetch_add(1, std::memory_order_relaxed);
    }
}

// Subsystem 3: Memory Arena Allocator & Thrashing Worker
void arena_thrash_worker(void* handle) {
    auto init = (arena_init_t)dlsym(handle, "TessArenaInit");
    auto alloc = (arena_alloc_t)dlsym(handle, "TessArenaAlloc");
    auto reset = (arena_reset_t)dlsym(handle, "TessArenaReset");
    auto free_arena = (arena_free_t)dlsym(handle, "TessArenaFree");
    if (!init || !alloc || !reset || !free_arena) return;

    void* arena = init(4 * 1024 * 1024); // 4MB Arena block
    if (!arena) return;

    while (test_active.load(std::memory_order_relaxed)) {
        for (int i = 0; i < 200; ++i) {
            alloc(arena, 64 + (i % 128));
        }
        reset(arena);
        ops_arena.fetch_add(1, std::memory_order_relaxed);
    }
    free_arena(arena);
}

int main(int argc, char* argv[]) {
    std::cout << "[*] Loading libTessSDK.so for Universal Max Stress Matrix...\n";
    
    // Load local shared library from repository files
    void* handle = dlopen("./libTessSDK.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "[!] Failed to load shared object: " << dlerror() << "\n";
        return 1;
    }

    auto q_create = (tess_mpmc_create_t)dlsym(handle, "tess_mpmc_queue_create");
    auto q_destroy = (tess_mpmc_destroy_t)dlsym(handle, "tess_mpmc_queue_destroy");
    void* shared_queue = q_create ? q_create(8192) : nullptr;

    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) cores = 4;
    
    int total_threads = cores * 3;
    std::cout << "[*] Spawning " << total_threads << " threads across " << cores << " hardware cores...\n";

    std::vector<std::thread> pool;
    for (int i = 0; i < total_threads; ++i) {
        if (i % 3 == 0) {
            pool.emplace_back(frame_pipeline_worker, handle);
        } else if (i % 3 == 1 && shared_queue) {
            pool.emplace_back(queue_contention_worker, handle, shared_queue);
        } else {
            pool.emplace_back(arena_thrash_worker, handle);
        }
    }

    int duration_sec = 60;
    std::cout << "[*] Stress test running at peak load for " << duration_sec << " seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));

    // Shutdown sequence
    test_active.store(false);
    for (auto& t : pool) {
        if (t.joinable()) t.join();
    }

    if (q_destroy && shared_queue) q_destroy(shared_queue);
    dlclose(handle);

    std::cout << "\n================ UNIVERSAL STRESS REPORT ================\n";
    std::cout << "Pipeline Frame Ops:       " << ops_math.load() << "\n";
    std::cout << "MPMC Queue Cycles:        " << ops_queue.load() << "\n";
    std::cout << "Arena Reset Iterations:   " << ops_arena.load() << "\n";
    std::cout << "System Faults / Errors:   " << system_faults.load() << "\n";
    std::cout << "=========================================================\n";

    return 0;
}
