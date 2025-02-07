// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#if defined(__x86_64__)
#include "distances_avx512.h"

#include <immintrin.h>

#include <cassert>
#include <cstdio>
#include <string>

#include "faiss/impl/platform_macros.h"
#include "utils_avx.h"
#include "utils_avx512.h"
#include "utils_sse.h"

namespace faiss {
namespace {
// reads 0 <= d < 4 floats as __m128
static inline __m128
masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

/// Function that does a component-wise operation between x and y
/// to compute L2 distances. ElementOp can then be used in the fvec_op_ny
/// functions below
struct ElementOpL2 {
    static float
    op(float x, float y) {
        float tmp = x - y;
        return tmp * tmp;
    }

    static __m128
    op(__m128 x, __m128 y) {
        __m128 tmp = _mm_sub_ps(x, y);
        return _mm_mul_ps(tmp, tmp);
    }

    static __m256
    op(__m256 x, __m256 y) {
        __m256 tmp = _mm256_sub_ps(x, y);
        return _mm256_mul_ps(tmp, tmp);
    }

    static __m512
    op(__m512 x, __m512 y) {
        __m512 tmp = _mm512_sub_ps(x, y);
        return _mm512_mul_ps(tmp, tmp);
    }
};

/// Function that does a component-wise operation between x and y
/// to compute inner products
struct ElementOpIP {
    static float
    op(float x, float y) {
        return x * y;
    }

    static __m128
    op(__m128 x, __m128 y) {
        return _mm_mul_ps(x, y);
    }

    static __m256
    op(__m256 x, __m256 y) {
        return _mm256_mul_ps(x, y);
    }

    static __m512
    op(__m512 x, __m512 y) {
        return _mm512_mul_ps(x, y);
    }
};

template <class ElementOp>
void
fvec_op_ny_D1(float* dis, const float* x, const float* y, size_t ny) {
    float x0s = x[0];
    __m128 x0 = _mm_set_ps(x0s, x0s, x0s, x0s);

    size_t i;
    for (i = 0; i + 3 < ny; i += 4) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        dis[i] = _mm_cvtss_f32(accu);
        __m128 tmp = _mm_shuffle_ps(accu, accu, 1);
        dis[i + 1] = _mm_cvtss_f32(tmp);
        tmp = _mm_shuffle_ps(accu, accu, 2);
        dis[i + 2] = _mm_cvtss_f32(tmp);
        tmp = _mm_shuffle_ps(accu, accu, 3);
        dis[i + 3] = _mm_cvtss_f32(tmp);
    }
    while (i < ny) {  // handle non-multiple-of-4 case
        dis[i++] = ElementOp::op(x0s, *y++);
    }
}

template <class ElementOp>
void
fvec_op_ny_D2(float* dis, const float* x, const float* y, size_t ny) {
    auto x0 = _mm512_set4_ps(x[1], x[0], x[1], x[0]);
    size_t i;
    for (i = 0; i + 15 < ny; i += 8) {
        auto accu = ElementOp::op(x0, _mm512_loadu_ps(y));
        y += 16;
        auto low = _mm512_extractf32x8_ps(accu, 0);
        auto high = _mm512_extractf32x8_ps(accu, 1);
        auto res = _mm256_hadd_ps(low, high);
        res = _mm256_permutevar8x32_ps(res, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        _mm256_storeu_ps(dis + i, res);
    }
    if (i < ny) {  // handle odd case
        dis[i] = ElementOp::op(x[0], y[0]) + ElementOp::op(x[1], y[1]);
    }
}

template <>
void
fvec_op_ny_D2<ElementOpIP>(float* dis, const float* x, const float* y, size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D2-vectors per loop.
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);

        for (i = 0; i < ny16 * 16; i += 16) {
            _mm_prefetch((const char*)(y + 64), _MM_HINT_T0);

            // load 16x2 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;

            transpose_16x2(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), v0, v1);

            // compute distances (dot product)
            __m512 distances = _mm512_mul_ps(m0, v0);
            distances = _mm512_fmadd_ps(m1, v1, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 32;  // move to the next set of 16x2 elements
        }
    }

    if (i < ny) {
        // process leftovers
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float distance = x0 * y[0] + x1 * y[1];
            y += 2;
            dis[i] = distance;
        }
    }
}

template <>
void
fvec_op_ny_D2<ElementOpL2>(float* dis, const float* x, const float* y, size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D2-vectors per loop.
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);

        for (i = 0; i < ny16 * 16; i += 16) {
            _mm_prefetch((const char*)(y + 64), _MM_HINT_T0);

            // load 16x2 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;

            transpose_16x2(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), v0, v1);

            // compute differences
            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);

            // compute squares of differences
            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 32;  // move to the next set of 16x2 elements
        }
    }

    if (i < ny) {
        // process leftovers
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float sub0 = x0 - y[0];
            float sub1 = x1 - y[1];
            float distance = sub0 * sub0 + sub1 * sub1;

            y += 2;
            dis[i] = distance;
        }
    }
}

template <class ElementOp>
void
fvec_op_ny_D4(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        dis[i] = horizontal_sum(accu);
    }
}

template <>
void
fvec_op_ny_D4<ElementOpIP>(float* dis, const float* x, const float* y, size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D4-vectors per loop.
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // load 16x4 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;

            transpose_16x4(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), _mm512_loadu_ps(y + 2 * 16),
                           _mm512_loadu_ps(y + 3 * 16), v0, v1, v2, v3);

            // compute distances
            __m512 distances = _mm512_mul_ps(m0, v0);
            distances = _mm512_fmadd_ps(m1, v1, distances);
            distances = _mm512_fmadd_ps(m2, v2, distances);
            distances = _mm512_fmadd_ps(m3, v3, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 64;  // move to the next set of 16x4 elements
        }
    }

    if (i < ny) {
        // process leftovers
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpIP::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void
fvec_op_ny_D4<ElementOpL2>(float* dis, const float* x, const float* y, size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D4-vectors per loop.
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // load 16x4 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;

            transpose_16x4(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), _mm512_loadu_ps(y + 2 * 16),
                           _mm512_loadu_ps(y + 3 * 16), v0, v1, v2, v3);

            // compute differences
            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);

            // compute squares of differences
            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 64;  // move to the next set of 16x4 elements
        }
    }

    if (i < ny) {
        // process leftovers
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpL2::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <class ElementOp>
void
fvec_op_ny_D8(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x1, _mm_loadu_ps(y)));
        y += 4;
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

template <>
void
fvec_op_ny_D8<ElementOpIP>(float* dis, const float* x, const float* y, size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D16-vectors per loop.
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);
        const __m512 m4 = _mm512_set1_ps(x[4]);
        const __m512 m5 = _mm512_set1_ps(x[5]);
        const __m512 m6 = _mm512_set1_ps(x[6]);
        const __m512 m7 = _mm512_set1_ps(x[7]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // load 16x8 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;
            __m512 v4;
            __m512 v5;
            __m512 v6;
            __m512 v7;

            transpose_16x8(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), _mm512_loadu_ps(y + 2 * 16),
                           _mm512_loadu_ps(y + 3 * 16), _mm512_loadu_ps(y + 4 * 16), _mm512_loadu_ps(y + 5 * 16),
                           _mm512_loadu_ps(y + 6 * 16), _mm512_loadu_ps(y + 7 * 16), v0, v1, v2, v3, v4, v5, v6, v7);

            // compute distances
            __m512 distances = _mm512_mul_ps(m0, v0);
            distances = _mm512_fmadd_ps(m1, v1, distances);
            distances = _mm512_fmadd_ps(m2, v2, distances);
            distances = _mm512_fmadd_ps(m3, v3, distances);
            distances = _mm512_fmadd_ps(m4, v4, distances);
            distances = _mm512_fmadd_ps(m5, v5, distances);
            distances = _mm512_fmadd_ps(m6, v6, distances);
            distances = _mm512_fmadd_ps(m7, v7, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 128;  // 16 floats * 8 rows
        }
    }

    if (i < ny) {
        // process leftovers
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpIP::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <>
void
fvec_op_ny_D8<ElementOpL2>(float* dis, const float* x, const float* y, size_t ny) {
    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // process 16 D16-vectors per loop.
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);
        const __m512 m4 = _mm512_set1_ps(x[4]);
        const __m512 m5 = _mm512_set1_ps(x[5]);
        const __m512 m6 = _mm512_set1_ps(x[6]);
        const __m512 m7 = _mm512_set1_ps(x[7]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // load 16x8 matrix and transpose it in registers.
            // the typical bottleneck is memory access, so
            // let's trade instructions for the bandwidth.

            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;
            __m512 v4;
            __m512 v5;
            __m512 v6;
            __m512 v7;

            transpose_16x8(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), _mm512_loadu_ps(y + 2 * 16),
                           _mm512_loadu_ps(y + 3 * 16), _mm512_loadu_ps(y + 4 * 16), _mm512_loadu_ps(y + 5 * 16),
                           _mm512_loadu_ps(y + 6 * 16), _mm512_loadu_ps(y + 7 * 16), v0, v1, v2, v3, v4, v5, v6, v7);

            // compute differences
            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);
            const __m512 d4 = _mm512_sub_ps(m4, v4);
            const __m512 d5 = _mm512_sub_ps(m5, v5);
            const __m512 d6 = _mm512_sub_ps(m6, v6);
            const __m512 d7 = _mm512_sub_ps(m7, v7);

            // compute squares of differences
            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);
            distances = _mm512_fmadd_ps(d4, d4, distances);
            distances = _mm512_fmadd_ps(d5, d5, distances);
            distances = _mm512_fmadd_ps(d6, d6, distances);
            distances = _mm512_fmadd_ps(d7, d7, distances);

            // store
            _mm512_storeu_ps(dis + i, distances);

            y += 128;  // 16 floats * 8 rows
        }
    }

    if (i < ny) {
        // process leftovers
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpL2::op(x0, _mm256_loadu_ps(y));
            y += 8;
            dis[i] = horizontal_sum(accu);
        }
    }
}

template <class ElementOp>
void
fvec_op_ny_D12(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);
    __m128 x2 = _mm_loadu_ps(x + 8);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x1, _mm_loadu_ps(y)));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x2, _mm_loadu_ps(y)));
        y += 4;
        dis[i] = horizontal_sum(accu);
    }
}

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
inline void
fvec_L2sqr_ny_avx512_impl(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    size_t i = 0;
    for (; i < ny; i += 4) {
        const float* __restrict y1 = y + d * i;
        const float* __restrict y2 = y + d * (i + 1);
        const float* __restrict y3 = y + d * (i + 2);
        const float* __restrict y4 = y + d * (i + 3);
        fvec_L2sqr_batch_4_avx512(x, y1, y2, y3, y4, d, dis[i], dis[i + 1], dis[i + 2], dis[i + 3]);
    }
    while (i < ny) {
        const float* __restrict y_i = y + d * i;
        dis[i] = fvec_L2sqr_avx512(x, y_i, d);
        y += d;
        i++;
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

size_t
fvec_L2sqr_ny_nearest_D2(float* distances_tmp_buffer, const float* x, const float* y, size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    size_t i = 0;
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    const size_t ny16 = ny / 16;
    if (ny16 > 0) {
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

        __m512 min_distances = _mm512_set1_ps(HUGE_VALF);
        __m512i min_indices = _mm512_set1_epi32(0);

        __m512i current_indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i indices_increment = _mm512_set1_epi32(16);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);

        for (; i < ny16 * 16; i += 16) {
            _mm_prefetch((const char*)(y + 64), _MM_HINT_T0);

            __m512 v0;
            __m512 v1;

            transpose_16x2(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), v0, v1);

            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);

            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);

            __mmask16 comparison = _mm512_cmp_ps_mask(distances, min_distances, _CMP_LT_OS);

            min_distances = _mm512_min_ps(distances, min_distances);
            min_indices = _mm512_mask_blend_epi32(comparison, min_indices, current_indices);

            current_indices = _mm512_add_epi32(current_indices, indices_increment);

            y += 32;
        }

        alignas(64) float min_distances_scalar[16];
        alignas(64) uint32_t min_indices_scalar[16];
        _mm512_store_ps(min_distances_scalar, min_distances);
        _mm512_store_epi32(min_indices_scalar, min_indices);

        for (size_t j = 0; j < 16; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float sub0 = x0 - y[0];
            float sub1 = x1 - y[1];
            float distance = sub0 * sub0 + sub1 * sub1;

            y += 2;

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}
size_t
fvec_L2sqr_ny_nearest_D4(float* distances_tmp_buffer, const float* x, const float* y, size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    size_t i = 0;
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    const size_t ny16 = ny / 16;

    if (ny16 > 0) {
        __m512 min_distances = _mm512_set1_ps(HUGE_VALF);
        __m512i min_indices = _mm512_set1_epi32(0);

        __m512i current_indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i indices_increment = _mm512_set1_epi32(16);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        for (; i < ny16 * 16; i += 16) {
            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;

            transpose_16x4(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), _mm512_loadu_ps(y + 2 * 16),
                           _mm512_loadu_ps(y + 3 * 16), v0, v1, v2, v3);

            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);

            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);

            __mmask16 comparison = _mm512_cmp_ps_mask(distances, min_distances, _CMP_LT_OS);

            min_distances = _mm512_min_ps(distances, min_distances);
            min_indices = _mm512_mask_blend_epi32(comparison, min_indices, current_indices);

            current_indices = _mm512_add_epi32(current_indices, indices_increment);

            y += 64;
        }

        alignas(64) float min_distances_scalar[16];
        alignas(64) uint32_t min_indices_scalar[16];
        _mm512_store_ps(min_distances_scalar, min_distances);
        _mm512_store_epi32(min_indices_scalar, min_indices);

        for (size_t j = 0; j < 16; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpL2::op(x0, _mm_loadu_ps(y));
            y += 4;
            const float distance = horizontal_sum(accu);

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}

size_t
fvec_L2sqr_ny_nearest_D8(float* distances_tmp_buffer, const float* x, const float* y, size_t ny) {
    // this implementation does not use distances_tmp_buffer.

    size_t i = 0;
    float current_min_distance = HUGE_VALF;
    size_t current_min_index = 0;

    const size_t ny16 = ny / 16;
    if (ny16 > 0) {
        __m512 min_distances = _mm512_set1_ps(HUGE_VALF);
        __m512i min_indices = _mm512_set1_epi32(0);

        __m512i current_indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512i indices_increment = _mm512_set1_epi32(16);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        const __m512 m4 = _mm512_set1_ps(x[4]);
        const __m512 m5 = _mm512_set1_ps(x[5]);
        const __m512 m6 = _mm512_set1_ps(x[6]);
        const __m512 m7 = _mm512_set1_ps(x[7]);

        for (; i < ny16 * 16; i += 16) {
            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;
            __m512 v4;
            __m512 v5;
            __m512 v6;
            __m512 v7;

            transpose_16x8(_mm512_loadu_ps(y + 0 * 16), _mm512_loadu_ps(y + 1 * 16), _mm512_loadu_ps(y + 2 * 16),
                           _mm512_loadu_ps(y + 3 * 16), _mm512_loadu_ps(y + 4 * 16), _mm512_loadu_ps(y + 5 * 16),
                           _mm512_loadu_ps(y + 6 * 16), _mm512_loadu_ps(y + 7 * 16), v0, v1, v2, v3, v4, v5, v6, v7);

            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);
            const __m512 d4 = _mm512_sub_ps(m4, v4);
            const __m512 d5 = _mm512_sub_ps(m5, v5);
            const __m512 d6 = _mm512_sub_ps(m6, v6);
            const __m512 d7 = _mm512_sub_ps(m7, v7);

            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);
            distances = _mm512_fmadd_ps(d4, d4, distances);
            distances = _mm512_fmadd_ps(d5, d5, distances);
            distances = _mm512_fmadd_ps(d6, d6, distances);
            distances = _mm512_fmadd_ps(d7, d7, distances);

            __mmask16 comparison = _mm512_cmp_ps_mask(distances, min_distances, _CMP_LT_OS);

            min_distances = _mm512_min_ps(distances, min_distances);
            min_indices = _mm512_mask_blend_epi32(comparison, min_indices, current_indices);

            current_indices = _mm512_add_epi32(current_indices, indices_increment);

            y += 128;
        }

        alignas(64) float min_distances_scalar[16];
        alignas(64) uint32_t min_indices_scalar[16];
        _mm512_store_ps(min_distances_scalar, min_distances);
        _mm512_store_epi32(min_indices_scalar, min_indices);

        for (size_t j = 0; j < 16; j++) {
            if (current_min_distance > min_distances_scalar[j]) {
                current_min_distance = min_distances_scalar[j];
                current_min_index = min_indices_scalar[j];
            }
        }
    }

    if (i < ny) {
        __m256 x0 = _mm256_loadu_ps(x);

        for (; i < ny; i++) {
            __m256 accu = ElementOpL2::op(x0, _mm256_loadu_ps(y));
            y += 8;
            const float distance = horizontal_sum(accu);

            if (current_min_distance > distance) {
                current_min_distance = distance;
                current_min_index = i;
            }
        }
    }

    return current_min_index;
}

size_t
fvec_L2sqr_ny_nearest_avx512_impl(float* distances_tmp_buffer, const float* x, const float* y, size_t d, size_t ny) {
    fvec_L2sqr_ny_avx512(distances_tmp_buffer, x, y, d, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

}  // namespace

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_avx512(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        res += x[i] * y[i];
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

float
fp16_vec_L2sqr_avx512(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y));
        auto mx_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(x + 16)));
        auto my_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(y + 16)));
        mx_0 = mx_0 - my_0;
        mx_1 = mx_1 - my_1;
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        y += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y));
        mx = mx - my;
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        y += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        auto my = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y));
        mx = _mm512_sub_ps(mx, my);
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

float
bf16_vec_L2sqr_avx512(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y));
        auto mx_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(x + 16)));
        auto my_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(y + 16)));
        mx_0 = mx_0 - my_0;
        mx_1 = mx_1 - my_1;
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        y += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y));
        mx = mx - my;
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        y += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        auto my = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y));
        mx = _mm512_sub_ps(mx, my);
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_avx512_bf16_patch(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        res += x[i] * bf16_float(y[i]);
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_L2sqr_avx512(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

float
fp16_vec_inner_product_avx512(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y));
        auto mx_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(x + 16)));
        auto my_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(y + 16)));
        m512_res = _mm512_fmadd_ps(mx_0, my_0, m512_res);
        m512_res = _mm512_fmadd_ps(mx_1, my_1, m512_res);
        x += 32;
        y += 32;
        d -= 32;
    }
    if (d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y));
        m512_res = _mm512_fmadd_ps(mx, my, m512_res);
        x += 16;
        y += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        auto my = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y));
        m512_res = _mm512_fmadd_ps(mx, my, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

float
bf16_vec_inner_product_avx512(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y));
        auto mx_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(x + 16)));
        auto my_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(y + 16)));
        m512_res = _mm512_fmadd_ps(mx_0, my_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, my_1, m512_res_0);
        x += 32;
        y += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y));
        m512_res = _mm512_fmadd_ps(mx, my, m512_res);
        x += 16;
        y += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        auto my = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y));
        m512_res = _mm512_fmadd_ps(mx, my, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_L2sqr_avx512_bf16_patch(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - bf16_float(y[i]);
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

float
fvec_L1_avx512(const float* x, const float* y, size_t d) {
    __m512 msum0 = _mm512_setzero_ps();
    __m512 signmask0 = __m512(_mm512_set1_epi32(0x7fffffffUL));

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps(x);
        x += 16;
        __m512 my = _mm512_loadu_ps(y);
        y += 16;
        const __m512 a_m_b = mx - my;
        msum0 += _mm512_and_ps(signmask0, a_m_b);
        d -= 16;
    }

    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 += _mm512_extractf32x8_ps(msum0, 0);
    __m256 signmask1 = __m256(_mm256_set1_epi32(0x7fffffffUL));

    if (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b = mx - my;
        msum1 += _mm256_and_ps(signmask1, a_m_b);
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 += _mm256_extractf128_ps(msum1, 0);
    __m128 signmask2 = __m128(_mm_set1_epi32(0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b = mx - my;
        msum2 += _mm_and_ps(signmask2, a_m_b);
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b = mx - my;
        msum2 += _mm_and_ps(signmask2, a_m_b);
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

float
fvec_Linf_avx512(const float* x, const float* y, size_t d) {
    __m512 msum0 = _mm512_setzero_ps();
    __m512 signmask0 = __m512(_mm512_set1_epi32(0x7fffffffUL));

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps(x);
        x += 16;
        __m512 my = _mm512_loadu_ps(y);
        y += 16;
        const __m512 a_m_b = mx - my;
        msum0 = _mm512_max_ps(msum0, _mm512_and_ps(signmask0, a_m_b));
        d -= 16;
    }

    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 = _mm256_max_ps(msum1, _mm512_extractf32x8_ps(msum0, 0));
    __m256 signmask1 = __m256(_mm256_set1_epi32(0x7fffffffUL));

    if (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b = mx - my;
        msum1 = _mm256_max_ps(msum1, _mm256_and_ps(signmask1, a_m_b));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_max_ps(msum2, _mm256_extractf128_ps(msum1, 0));
    __m128 signmask2 = __m128(_mm_set1_epi32(0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b = mx - my;
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b = mx - my;
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
    }

    msum2 = _mm_max_ps(_mm_movehl_ps(msum2, msum2), msum2);
    msum2 = _mm_max_ps(msum2, _mm_shuffle_ps(msum2, msum2, 1));
    return _mm_cvtss_f32(msum2);
}

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_madd_avx512(size_t n, const float* a, float bf, const float* b, float* c) {
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_inner_product_batch_4_avx512(const float* __restrict x, const float* __restrict y0, const float* __restrict y1,
                                  const float* __restrict y2, const float* __restrict y3, const size_t d, float& dis0,
                                  float& dis1, float& dis2, float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        d0 += x[i] * y0[i];
        d1 += x[i] * y1[i];
        d2 += x[i] * y2[i];
        d3 += x[i] * y3[i];
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_inner_product_batch_4_avx512_bf16_patch(const float* __restrict x, const float* __restrict y0,
                                             const float* __restrict y1, const float* __restrict y2,
                                             const float* __restrict y3, const size_t d, float& dis0, float& dis1,
                                             float& dis2, float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        d0 += x[i] * bf16_float(y0[i]);
        d1 += x[i] * bf16_float(y1[i]);
        d2 += x[i] * bf16_float(y2[i]);
        d3 += x[i] * bf16_float(y3[i]);
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

void
fp16_vec_inner_product_batch_4_avx512(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                      const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                                      float& dis1, float& dis2, float& dis3) {
    __m512 m512_res_0 = _mm512_setzero_ps();
    __m512 m512_res_1 = _mm512_setzero_ps();
    __m512 m512_res_2 = _mm512_setzero_ps();
    __m512 m512_res_3 = _mm512_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y0));
        auto my1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y1));
        auto my2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y2));
        auto my3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y3));
        m512_res_0 = _mm512_fmadd_ps(mx, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(mx, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(mx, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(mx, my3, m512_res_3);
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
        cur_d -= 16;
    }
    if (cur_d > 0) {
        const __mmask16 mask = (1U << cur_d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        auto my0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y0));
        auto my1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y1));
        auto my2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y2));
        auto my3 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y3));
        m512_res_0 = _mm512_fmadd_ps(mx, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(mx, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(mx, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(mx, my3, m512_res_3);
    }
    dis0 = _mm512_reduce_add_ps(m512_res_0);
    dis1 = _mm512_reduce_add_ps(m512_res_1);
    dis2 = _mm512_reduce_add_ps(m512_res_2);
    dis3 = _mm512_reduce_add_ps(m512_res_3);
    return;
}

void
bf16_vec_inner_product_batch_4_avx512(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                      const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                      float& dis1, float& dis2, float& dis3) {
    __m512 m512_res_0 = _mm512_setzero_ps();
    __m512 m512_res_1 = _mm512_setzero_ps();
    __m512 m512_res_2 = _mm512_setzero_ps();
    __m512 m512_res_3 = _mm512_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y0));
        auto my1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y1));
        auto my2 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y2));
        auto my3 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y3));
        m512_res_0 = _mm512_fmadd_ps(mx, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(mx, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(mx, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(mx, my3, m512_res_3);
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
        cur_d -= 16;
    }
    if (cur_d > 0) {
        const __mmask16 mask = (1U << cur_d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        auto my0 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y0));
        auto my1 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y1));
        auto my2 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y2));
        auto my3 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y3));
        m512_res_0 = _mm512_fmadd_ps(mx, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(mx, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(mx, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(mx, my3, m512_res_3);
    }
    dis0 = _mm512_reduce_add_ps(m512_res_0);
    dis1 = _mm512_reduce_add_ps(m512_res_1);
    dis2 = _mm512_reduce_add_ps(m512_res_2);
    dis3 = _mm512_reduce_add_ps(m512_res_3);
    return;
}

void
fp16_vec_L2sqr_batch_4_avx512(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                              const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                              float& dis1, float& dis2, float& dis3) {
    __m512 m512_res_0 = _mm512_setzero_ps();
    __m512 m512_res_1 = _mm512_setzero_ps();
    __m512 m512_res_2 = _mm512_setzero_ps();
    __m512 m512_res_3 = _mm512_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y0));
        auto my1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y1));
        auto my2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y2));
        auto my3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y3));
        my0 = _mm512_sub_ps(mx, my0);
        my1 = _mm512_sub_ps(mx, my1);
        my2 = _mm512_sub_ps(mx, my2);
        my3 = _mm512_sub_ps(mx, my3);
        m512_res_0 = _mm512_fmadd_ps(my0, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(my1, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(my2, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(my3, my3, m512_res_3);
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
        cur_d -= 16;
    }
    if (cur_d > 0) {
        const __mmask16 mask = (1U << cur_d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        auto my0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y0));
        auto my1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y1));
        auto my2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y2));
        auto my3 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y3));
        my0 = _mm512_sub_ps(mx, my0);
        my1 = _mm512_sub_ps(mx, my1);
        my2 = _mm512_sub_ps(mx, my2);
        my3 = _mm512_sub_ps(mx, my3);
        m512_res_0 = _mm512_fmadd_ps(my0, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(my1, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(my2, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(my3, my3, m512_res_3);
    }
    dis0 = _mm512_reduce_add_ps(m512_res_0);
    dis1 = _mm512_reduce_add_ps(m512_res_1);
    dis2 = _mm512_reduce_add_ps(m512_res_2);
    dis3 = _mm512_reduce_add_ps(m512_res_3);
    return;
}

void
bf16_vec_L2sqr_batch_4_avx512(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                              const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                              float& dis1, float& dis2, float& dis3) {
    __m512 m512_res_0 = _mm512_setzero_ps();
    __m512 m512_res_1 = _mm512_setzero_ps();
    __m512 m512_res_2 = _mm512_setzero_ps();
    __m512 m512_res_3 = _mm512_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y0));
        auto my1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y1));
        auto my2 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y2));
        auto my3 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y3));
        my0 = mx - my0;
        my1 = mx - my1;
        my2 = mx - my2;
        my3 = mx - my3;
        m512_res_0 = _mm512_fmadd_ps(my0, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(my1, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(my2, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(my3, my3, m512_res_3);
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
        cur_d -= 16;
    }
    if (cur_d > 0) {
        const __mmask16 mask = (1U << cur_d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        auto my0 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y0));
        auto my1 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y1));
        auto my2 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y2));
        auto my3 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y3));
        my0 = _mm512_sub_ps(mx, my0);
        my1 = _mm512_sub_ps(mx, my1);
        my2 = _mm512_sub_ps(mx, my2);
        my3 = _mm512_sub_ps(mx, my3);
        m512_res_0 = _mm512_fmadd_ps(my0, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(my1, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(my2, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(my3, my3, m512_res_3);
    }
    dis0 = _mm512_reduce_add_ps(m512_res_0);
    dis1 = _mm512_reduce_add_ps(m512_res_1);
    dis2 = _mm512_reduce_add_ps(m512_res_2);
    dis3 = _mm512_reduce_add_ps(m512_res_3);
    return;
}
// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_L2sqr_batch_4_avx512(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                          const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_L2sqr_batch_4_avx512_bf16_patch(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                     const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - bf16_float(y0[i]);
        const float q1 = x[i] - bf16_float(y1[i]);
        const float q2 = x[i] - bf16_float(y2[i]);
        const float q3 = x[i] - bf16_float(y3[i]);
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
int32_t
ivec_inner_product_avx512(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;
    for (i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

// trust the compiler to unroll this properly
int32_t
ivec_L2sqr_avx512(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;
    for (i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return res;
}

float
fvec_norm_L2sqr_avx512(const float* x, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_loadu_ps(x);
        auto mx_1 = _mm512_loadu_ps(x + 16);
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_loadu_ps(x);
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_maskz_loadu_ps(mask, x);
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

float
fp16_vec_norm_L2sqr_avx512(const knowhere::fp16* x, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto mx_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(x + 16)));
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

float
bf16_vec_norm_L2sqr_avx512(const knowhere::bf16* x, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto mx_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(x + 16)));
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

void
fvec_L2sqr_ny_avx512(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    // optimized for a few special cases

#define DISPATCH(dval)                                  \
    case dval:                                          \
        fvec_op_ny_D##dval<ElementOpL2>(dis, x, y, ny); \
        return;

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        default:
            fvec_L2sqr_ny_avx512_impl(dis, x, y, d, ny);
            return;
    }
#undef DISPATCH
}

size_t
fvec_L2sqr_ny_nearest_avx512(float* distances_tmp_buffer, const float* x, const float* y, size_t d, size_t ny) {
// optimized for a few special cases
#define DISPATCH(dval) \
    case dval:         \
        return fvec_L2sqr_ny_nearest_D##dval(distances_tmp_buffer, x, y, ny);

    switch (d) {
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        default:
            return fvec_L2sqr_ny_nearest_avx512_impl(distances_tmp_buffer, x, y, d, ny);
    }
#undef DISPATCH
}

}  // namespace faiss

#endif
