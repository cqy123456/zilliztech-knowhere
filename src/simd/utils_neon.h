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

#ifndef UTILS_NEON_H
#define UTILS_NEON_H
#include <cassert>

#include "knowhere/operands.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace faiss {
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
#endif /* UTILS_NEON_H */
