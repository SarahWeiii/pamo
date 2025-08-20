#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH
#define STACK_SIZE 256
#include "predicator.cuh"

namespace lbvh
{
    template <typename Real, typename Objects, bool IsConst, typename OutputIterator>
    __device__ unsigned int get_number_of_intersect_candidates(
        const detail::basic_device_bvh<Real, Objects, IsConst> &bvh,
        const query_overlap<Real> q, OutputIterator outiter,
        const unsigned int query_idx,
        const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        index_type stack[STACK_SIZE];
        index_type *stack_ptr = stack;
        OutputIterator outiter_begin = outiter;
        *stack_ptr++ = 0; // root node is always 0

        unsigned int num_found = 0;
        do
        {
            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            if (intersects(q.target, bvh.aabbs[L_idx]))
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if(obj_idx != query_idx)
                    {
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = L_idx;
                }
            }
            if (intersects(q.target, bvh.aabbs[R_idx]))
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if(obj_idx != query_idx)
                    {
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = R_idx;
                }
            }

        } while (stack < stack_ptr);
        return num_found;
    }



    // dfs code ----------------------
    template <typename Real, typename Objects, bool IsConst, typename OutputIterator>
    __device__ unsigned int query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst> &bvh,
        const query_overlap<Real> q, OutputIterator outiter,
        const unsigned int query_idx,
        const unsigned int first = 0xFFFFFFFF) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        index_type stack[STACK_SIZE]; // is it okay?
        index_type *stack_ptr = stack;
        OutputIterator outiter_begin = outiter;
        *stack_ptr++ = 0; // root node is always 0
        unsigned int num_found = 0;
        int iter = 0;

        // dynamic buffer
        outiter += first;
        do
        {
            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            if (intersects(q.target, bvh.aabbs[L_idx]))
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if(obj_idx != query_idx)
                    {
                        *outiter++ = query_idx;
                        *outiter++ = obj_idx;
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = L_idx;
                }
            }
            if (intersects(q.target, bvh.aabbs[R_idx]))
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if(obj_idx != query_idx)
                    {
                        *outiter++ = query_idx;
                        *outiter++ = obj_idx;
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = R_idx;
                }
            }
        } while (stack < stack_ptr);
        return num_found;
    }

}

#endif // LBVH_QUERY_CUH