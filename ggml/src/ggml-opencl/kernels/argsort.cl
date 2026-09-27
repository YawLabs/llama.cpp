#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define SWAP(x, y, T) { T tmp = (x); (x) = (y); (y) = tmp; }

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

kernel void kernel_argsort_f32_i32(
    global float * src0,
    ulong          offset0,
    global int   * dst,
    ulong          offsetd,
    const int      ne00,
    const int      ne00_pad,
    const int      order,
    local int    * dst_row
) {
    // bitonic sort
    int col = get_local_id(0);
    int row = get_group_id(1);

    if (col >= ne00_pad) {
        return;
    }

    src0 = (global char  *)((global char *)src0 + offset0);
    dst  = (global float *)((global char *)dst  + offsetd);

    global float * x_row = src0 + row * ne00;

    // initialize indices
    dst_row[col] = col;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= ne00_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ne00 ||
                        (dst_row[ixj] < ne00 && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj], int);
                    }
                } else {
                    if (dst_row[ixj] >= ne00 ||
                        (dst_row[col] < ne00 && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj], int);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // copy the result to dst without the padding
    if (col < ne00) {
        dst[row * ne00 + col] = dst_row[col];
    }
}

//------------------------------------------------------------------------------
// Argsort and top-k for rows of any length: a bitonic network over each row padded
// to npad, a power of two. kernel_argsort_chunk_f32_i32 runs the passes with a
// stride below its chunk (2 x local size) in local memory; the passes with a
// longer stride run one launch each in kernel_argsort_step_f32_i32, over the
// idx/val scratch in global memory (npad entries per row). Padding (idx >= ne00)
// sorts last, and equal values sort by index, so the order is deterministic.
//------------------------------------------------------------------------------

// true when (va, ia) goes before (vb, ib)
inline bool argsort_before(float va, int ia, float vb, int ib, int ne00, int order) {
    if (ia >= ne00) {
        return false;
    }
    if (ib >= ne00) {
        return true;
    }
    if (va != vb) {
        return order == GGML_SORT_ORDER_ASC ? va < vb : va > vb;
    }
    return ia < ib;
}

// k_lo == 2: load the row chunk from src0 and run the stages k = 2 .. k_hi.
// k_lo > 2: load the chunk from the scratch and finish stage k_lo (k_lo == k_hi).
// k_hi == npad: the row is sorted, write its first n_out indices to dst; else write the scratch.
kernel void kernel_argsort_chunk_f32_i32(
    global char  * src0,
    ulong          offset0,
    global int   * sidx,
    global float * sval,
    global int   * dst,
    ulong          offsetd,
    const int      ne00,
    const int      npad,
    const int      n_out,
    const int      ne01,
    const int      ne02,
    const ulong    nb01,
    const ulong    nb02,
    const ulong    nb03,
    const int      order,
    const int      k_lo,
    const int      k_hi,
    local int    * li,
    local float  * lv
) {
    const int lid  = get_local_id(0);
    const int nth  = get_local_size(0);
    const int cs   = 2*nth;
    const int base = get_group_id(0)*cs;
    const int row  = get_group_id(1);

    global int   * ri = sidx + (ulong)row*npad;
    global float * rv = sval + (ulong)row*npad;

    if (k_lo == 2) {
        const int i01 = row % ne01;
        const int i02 = (row / ne01) % ne02;
        const int i03 = row / (ne01*ne02);
        global const float * x = (global const float *)(src0 + offset0 + i01*nb01 + i02*nb02 + i03*nb03);
        for (int t = lid; t < cs; t += nth) {
            const int g = base + t;
            li[t] = g;
            lv[t] = g < ne00 ? x[g] : 0.0f;
        }
    } else {
        for (int t = lid; t < cs; t += nth) {
            li[t] = ri[base + t];
            lv[t] = rv[base + t];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = k_lo; k <= k_hi; k *= 2) {
        for (int j = min(k, cs)/2; j > 0; j /= 2) {
            const int   lo  = ((lid & ~(j - 1)) << 1) | (lid & (j - 1));
            const int   hi  = lo + j;
            const bool  asc = ((base + lo) & k) == 0;
            const int   ia  = li[lo];
            const int   ib  = li[hi];
            const float va  = lv[lo];
            const float vb  = lv[hi];
            if (asc ? argsort_before(vb, ib, va, ia, ne00, order) : argsort_before(va, ia, vb, ib, ne00, order)) {
                li[lo] = ib;
                li[hi] = ia;
                lv[lo] = vb;
                lv[hi] = va;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (k_hi == npad) {
        global int * d = (global int *)((global char *)dst + offsetd) + (ulong)row*n_out;
        for (int t = lid; t < cs; t += nth) {
            if (base + t < n_out) {
                d[base + t] = li[t];
            }
        }
    } else {
        for (int t = lid; t < cs; t += nth) {
            ri[base + t] = li[t];
            rv[base + t] = lv[t];
        }
    }
}

// one compare-exchange per work item: pass (k, j) of the network, j >= the chunk size
kernel void kernel_argsort_step_f32_i32(
    global int   * sidx,
    global float * sval,
    const int      ne00,
    const int      npad,
    const int      order,
    const int      k,
    const int      j
) {
    const int p   = get_global_id(0);
    const int row = get_global_id(1);

    global int   * ri = sidx + (ulong)row*npad;
    global float * rv = sval + (ulong)row*npad;

    const int   lo  = ((p & ~(j - 1)) << 1) | (p & (j - 1));
    const int   hi  = lo + j;
    const bool  asc = (lo & k) == 0;
    const int   ia  = ri[lo];
    const int   ib  = ri[hi];
    const float va  = rv[lo];
    const float vb  = rv[hi];
    if (asc ? argsort_before(vb, ib, va, ia, ne00, order) : argsort_before(va, ia, vb, ib, ne00, order)) {
        ri[lo] = ib;
        ri[hi] = ia;
        rv[lo] = vb;
        rv[hi] = va;
    }
}

// One pass of top-k by reduction: each workgroup sorts a chunk of 2 x local size entries of its row
// in local memory, descending, and keeps the first kpad (a power of two >= k, at most half a chunk).
// The first pass reads the row from src0, later passes the (idx, val) pairs the pass before kept.
// The last pass has one chunk per row and writes its first k indices to dst.
kernel void kernel_top_k_chunk_f32_i32(
    global char  * src0,
    ulong          offset0,
    global int   * in_idx,
    global float * in_val,
    global int   * out_idx,
    global float * out_val,
    global int   * dst,
    ulong          offsetd,
    const int      ne00,
    const int      n_in,
    const int      in_stride,
    const int      out_stride,
    const int      k,
    const int      kpad,
    const int      ne01,
    const int      ne02,
    const ulong    nb01,
    const ulong    nb02,
    const ulong    nb03,
    const int      first,
    const int      last,
    local int    * li,
    local float  * lv
) {
    const int lid   = get_local_id(0);
    const int nth   = get_local_size(0);
    const int cs    = 2*nth;
    const int chunk = get_group_id(0);
    const int base  = chunk*cs;
    const int row   = get_group_id(1);

    if (first) {
        const int i01 = row % ne01;
        const int i02 = (row / ne01) % ne02;
        const int i03 = row / (ne01*ne02);
        global const float * x = (global const float *)(src0 + offset0 + i01*nb01 + i02*nb02 + i03*nb03);
        for (int t = lid; t < cs; t += nth) {
            const int g = base + t;
            li[t] = g < n_in ? g    : INT_MAX;
            lv[t] = g < n_in ? x[g] : 0.0f;
        }
    } else {
        global const int   * ri = in_idx + (ulong)row*in_stride;
        global const float * rv = in_val + (ulong)row*in_stride;
        for (int t = lid; t < cs; t += nth) {
            const int g = base + t;
            li[t] = g < n_in ? ri[g] : INT_MAX;
            lv[t] = g < n_in ? rv[g] : 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k2 = 2; k2 <= cs; k2 *= 2) {
        for (int j = k2/2; j > 0; j /= 2) {
            const int   lo  = ((lid & ~(j - 1)) << 1) | (lid & (j - 1));
            const int   hi  = lo + j;
            const bool  asc = (lo & k2) == 0;
            const int   ia  = li[lo];
            const int   ib  = li[hi];
            const float va  = lv[lo];
            const float vb  = lv[hi];
            if (asc ? argsort_before(vb, ib, va, ia, ne00, GGML_SORT_ORDER_DESC) : argsort_before(va, ia, vb, ib, ne00, GGML_SORT_ORDER_DESC)) {
                li[lo] = ib;
                li[hi] = ia;
                lv[lo] = vb;
                lv[hi] = va;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (last) {
        global int * d = (global int *)((global char *)dst + offsetd) + (ulong)row*k;
        for (int t = lid; t < k; t += nth) {
            d[t] = li[t];
        }
    } else {
        global int   * ri = out_idx + (ulong)row*out_stride + chunk*kpad;
        global float * rv = out_val + (ulong)row*out_stride + chunk*kpad;
        for (int t = lid; t < kpad; t += nth) {
            ri[t] = li[t];
            rv[t] = lv[t];
        }
    }
}
