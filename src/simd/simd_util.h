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

#ifndef SIMD_UTIL_H
#define SIMD_UTIL_H
#include <cassert>

#include "knowhere/operands.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#endif
namespace faiss {
#if defined(__x86_64__)
#define ALIGNED(x) __attribute__((aligned(x)))

static inline __m128
_mm_bf16_to_fp32(const __m128i& a) {
    auto o = _mm_slli_epi32(_mm_cvtepu16_epi32(a), 16);
    return _mm_castsi128_ps(o);
}

static inline __m256
_mm256_bf16_to_fp32(const __m128i& a) {
    __m256i o = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a), 16);
    return _mm256_castsi256_ps(o);
}

static inline __m512
_mm512_bf16_to_fp32(const __m256i& x) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(x), 16));
}

static inline __m128i
mm_masked_read_short(int d, const uint16_t* x) {
    assert(0 <= d && d < 8);
    ALIGNED(16) uint16_t buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    switch (d) {
        case 7:
            buf[6] = x[6];
        case 6:
            buf[5] = x[5];
        case 5:
            buf[4] = x[4];
        case 4:
            buf[3] = x[3];
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_loadu_si128((__m128i*)buf);
}

static inline float
_mm256_reduce_add_ps(const __m256 res) {
    const __m128 sum = _mm_add_ps(_mm256_castps256_ps128(res), _mm256_extractf128_ps(res, 1));
    const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
    const __m128 v1 = _mm_add_ps(sum, v0);
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    const __m128 v3 = _mm_add_ps(v1, v2);
    return _mm_cvtss_f32(v3);
}

static inline
void fvec_L2sqr_ny_dim4(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    size_t y_i = ny;
    auto mx_t = _mm_loadu_ps(x);
    auto mx = _mm256_set_m128(mx_t, mx_t);
    const float* y_i_addr = y;
    while (y_i >= 8) {
        auto my1 = _mm256_loadu_ps(y_i_addr);
        auto my2 = _mm256_loadu_ps(y_i_addr + 2 * d);
        auto my3 = _mm256_loadu_ps(y_i_addr + 4 * d);
        auto my4 = _mm256_loadu_ps(y_i_addr + 6 * d);
        my1 = _mm256_sub_ps(my1, mx);
        my1 = _mm256_mul_ps(my1, my1); 
        my2 = _mm256_sub_ps(my2, mx);
        my2 = _mm256_mul_ps(my2, my2); 
        my3 = _mm256_sub_ps(my3, mx);
        my3 = _mm256_mul_ps(my3, my3); 
        my4 = _mm256_sub_ps(my4, mx);
        my4 = _mm256_mul_ps(my4, my4); 
        my1 = _mm256_hadd_ps(my1, my2);
        my3 = _mm256_hadd_ps(my3, my4);
        my1 = _mm256_hadd_ps(my1, my3);
        my1 = _mm256_permutevar8x32_ps(my1, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_ps(dis + (ny - y_i), my1);
        y_i_addr += 8 * d;
        y_i -=8;
    }
    while (y_i >= 4) {
        auto my1 = _mm256_loadu_ps(y_i_addr);
        y_i_addr = y_i_addr + 2 * d;
        auto my2 = _mm256_loadu_ps(y_i_addr);
        y_i_addr = y_i_addr + 2 * d;
        my1 = _mm256_sub_ps(my1, mx);
        my1 = _mm256_mul_ps(my1, my1); 
        my2 = _mm256_sub_ps(my2, mx);
        my2 = _mm256_mul_ps(my2, my2); 
        my1 = _mm256_hadd_ps(my1, my2);
        my1 = _mm256_hadd_ps(my1, my1);
         my1 = _mm256_permutevar8x32_ps(my1, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        __m128 res = _mm256_extractf128_ps(my1, 0);
        _mm_storeu_ps(dis + (ny - y_i), res);
        y_i -=4;
    }
    if (y_i >=2) {
        auto my1 = _mm256_loadu_ps(y_i_addr);
        y_i_addr = y_i_addr + 2 * d;
        my1 = _mm256_sub_ps(my1, mx);
        my1 = _mm256_mul_ps(my1, my1); 
        __m128 high = _mm256_extractf128_ps(my1, 1);
        __m128 low = _mm256_extractf128_ps(my1, 0);
        __m128 sum_low = _mm_hadd_ps(low, low);
        sum_low = _mm_hadd_ps(sum_low, sum_low);

        __m128 sum_high = _mm_hadd_ps(high, high);
        sum_high = _mm_hadd_ps(sum_high, sum_high);

        dis[ny - y_i] = _mm_cvtss_f32(sum_low);
        dis[ny - y_i + 1] = _mm_cvtss_f32(sum_high);
        y_i -=2;
    }
    if (y_i >=0) {
        float dis1, dis2;
        dis1 = (x[0] - y_i_addr[0]) * (x[0] - y_i_addr[0]);
        dis2 = (x[1] - y_i_addr[1]) * (x[1] - y_i_addr[1]);
        dis1 += (x[2] - y_i_addr[2]) * (x[2] - y_i_addr[2]);
        dis2 += (x[3] - y_i_addr[3]) * (x[3] - y_i_addr[3]);
        dis[ny - y_i] = dis1 + dis2;
    }
}

#endif

#if defined(__ARM_NEON)
static inline float32x4x4_t
vcvt4_f32_f16(const float16x4x4_t a) {
    float32x4x4_t c;
    c.val[0] = vcvt_f32_f16(a.val[0]);
    c.val[1] = vcvt_f32_f16(a.val[1]);
    c.val[2] = vcvt_f32_f16(a.val[2]);
    c.val[3] = vcvt_f32_f16(a.val[3]);
    return c;
}

static inline float32x4x2_t
vcvt2_f32_f16(const float16x4x2_t a) {
    float32x4x2_t c;
    c.val[0] = vcvt_f32_f16(a.val[0]);
    c.val[1] = vcvt_f32_f16(a.val[1]);
    return c;
}

static inline float32x4x4_t
vcvt4_f32_half(const uint16x4x4_t x) {
    float32x4x4_t c;
    c.val[0] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[0]), 16));
    c.val[1] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[1]), 16));
    c.val[2] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[2]), 16));
    c.val[3] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[3]), 16));
    return c;
}

static inline float32x4x2_t
vcvt2_f32_half(const uint16x4x2_t x) {
    float32x4x2_t c;
    c.val[0] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[0]), 16));
    c.val[1] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[1]), 16));
    return c;
}

static inline float32x4_t
vcvt_f32_half(const uint16x4_t x) {
    return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x), 16));
}

#endif
}  // namespace faiss
#endif /* SIMD_UTIL_H */
