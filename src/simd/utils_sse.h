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

#ifndef UTILS_SSE_H
#define UTILS_SSE_H
#include <cassert>

#include "knowhere/operands.h"

#if defined(__x86_64__)
#include <immintrin.h>

namespace faiss {
#define ALIGNED(x) __attribute__((aligned(x)))

static inline __m128
_mm_bf16_to_fp32(const __m128i& a) {
    auto o = _mm_slli_epi32(_mm_cvtepu16_epi32(a), 16);
    return _mm_castsi128_ps(o);
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
horizontal_sum(const __m128 v) {
    // say, v is [x0, x1, x2, x3]

    // v0 is [x2, x3, ..., ...]
    const __m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));
    // v1 is [x0 + x2, x1 + x3, ..., ...]
    const __m128 v1 = _mm_add_ps(v, v0);
    // v2 is [x1 + x3, ..., .... ,...]
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    // v3 is [x0 + x1 + x2 + x3, ..., ..., ...]
    const __m128 v3 = _mm_add_ps(v1, v2);
    // return v3[0]
    return _mm_cvtss_f32(v3);
}

}  // namespace faiss
#endif
#endif /* UTILS_SSE_H */
