// Row-wise argmax of f32 rows into i32. One workgroup per row; the local size is a power of two.
// Ties pick the last index, as the CPU backend does.
kernel void kernel_argmax_f32(
    global char *  src0,
    ulong          offset0,
    global char *  dst,
    ulong          offsetd,
    int            ne00,
    ulong          nb01,
    local float *  lval,
    local int *    lidx
) {
    src0 = src0 + offset0;
    dst  = dst  + offsetd;

    const int i1    = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);

    global const float * src_row = (global const float *) (src0 + i1*nb01);

    float best = -INFINITY;
    int   idx  = 0;
    for (int i0 = lid; i0 < ne00; i0 += lsize) {
        const float v = src_row[i0];
        if (v >= best) {
            best = v;
            idx  = i0;
        }
    }

    lval[lid] = best;
    lidx[lid] = idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsize/2; s > 0; s >>= 1) {
        if (lid < s) {
            const float v = lval[lid + s];
            const int   j = lidx[lid + s];
            if (v > lval[lid] || (v == lval[lid] && j > lidx[lid])) {
                lval[lid] = v;
                lidx[lid] = j;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        ((global int *) dst)[i1] = lidx[0];
    }
}
