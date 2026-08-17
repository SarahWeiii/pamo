#pragma once
#include "commons.cuh"
#include <vector>
#include <map>

class MemoryAllocator
{
    std::map<void*, int64_t> sizesAlloc;
    std::map<void*, int64_t> sizesFree;
    size_t mapSize, maxFree;
public:
    MemoryAllocator(size_t map_size, size_t max_free)
    {
        mapSize = map_size;
        maxFree = max_free;
    }
    template<typename T>
    T * alloc(const size_t num)
    {
        size_t sz = num * sizeof(T);
        void* block = find_free_block(sz);
        if (block)
        {
            sizesAlloc[block] = sizesFree[block];
            sizesFree.erase(block);
            // printf("HIT %p %d\n", block, sz);
            return (T*)block;
        }
        sz = ceil_div(sz, mapSize) * mapSize;
        T * ptr;
        CHECK_CUDA(cudaMalloc(&ptr, sz));
        // printf("SHORT %p %d\n", ptr, sz);
        sizesAlloc[ptr] = sz;
        return ptr;
    }

    template<typename T>
    int64_t probe(void * ptr)
    {
        assert(sizesAlloc.count(ptr));
        return sizesAlloc[ptr] / sizeof(T);
    }

    void* find_free_block(size_t sz)
    {
        size_t minsize = PTRDIFF_MAX;
        void* got = nullptr;
        for (const auto& p : sizesFree)
        {
            if (p.second <= minsize && p.second >= sz)
            {
                got = p.first;
                minsize = p.second;
            }
        }
        return got;
    }

    void release_smallest()
    {
        if (sizesFree.size() == 0) return;
        void* tofree = find_free_block(0);
        assert(tofree != nullptr);
        CHECK_CUDA(cudaFree(tofree));
        // printf("RELEASE %p %d\n", tofree, sizesFree[tofree]);
        sizesFree.erase(tofree);
    }

    void free(void* ptr)
    {
        assert(sizesAlloc.count(ptr));
        sizesFree[ptr] = sizesAlloc[ptr];
        sizesAlloc.erase(ptr);
        // printf("FREE %p %d\n", ptr, sizesFree[ptr]);
        while (sizesFree.size() > maxFree)
            release_smallest();
    }

    void clear()
    {
        for (const auto& p : sizesAlloc)
            CHECK_CUDA(cudaFree(p.first));
        for (const auto& p : sizesFree)
            CHECK_CUDA(cudaFree(p.first));
        sizesAlloc.clear();
        sizesFree.clear();
    }

    virtual ~MemoryAllocator()
    {
        clear();
    }
};
