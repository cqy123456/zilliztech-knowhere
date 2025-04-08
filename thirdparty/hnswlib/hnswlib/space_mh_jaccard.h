#pragma once

#include <faiss/utils/binary_distances.h>

#include "hnswlib.h"

namespace hnswlib {


static float
MH_Jaccard(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -1.0 * faiss::fvec_minhash_jaccard((const float*)pVect1v, (const float*)pVect2v, *((size_t*)qty_ptr));
}

class MHJaccardSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    MHJaccardSpace(size_t dim) {
        fstdistfunc_ = MH_Jaccard;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<float>
    get_dist_func() {
        return fstdistfunc_;
    }

    void*
    get_dist_func_param() {
        return &dim_;
    }

    ~MHJaccardSpace() {
    }
};

}  // namespace hnswlib
