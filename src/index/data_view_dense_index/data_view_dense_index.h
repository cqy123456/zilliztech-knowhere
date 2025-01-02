// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// knowhere-specific indices
#pragma once

#include <faiss/impl/DistanceComputer.h>
#include <knowhere/bitsetview.h>
#include <knowhere/range_util.h>

#include <atomic>
#include <functional>
#include <memory>

#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/ResultHandler.h"
#include "faiss/utils/distances_if.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/operands.h"
#include "knowhere/range_util.h"
#include "knowhere/config.h"
#include "simd/hook.h"
#include "knowhere/utils.h"
#include <mutex>
#include <shared_mutex>
namespace knowhere {
using ViewDataOp = std::function<void*(size_t)>;
using idx_t = faiss::idx_t;
using CMAX = faiss::CMax<float, idx_t>;
using CMIN = faiss::CMin<float, idx_t>;
struct RangeSearchResult;
struct DataViewIndexBase {
    int d;
    DataFormatEnum data_type;
    const ViewDataOp& view_data;
    MetricType metric_type;

    int code_size;
    std::atomic<idx_t> ntotal;
    bool is_cosine;
    std::vector<float> norms;
    mutable std::shared_mutex  norms_mutex_;

    explicit DataViewIndexBase(idx_t d, DataFormatEnum data_type, MetricType meteic_type,
                         const ViewDataOp& view)
        : d(d),
          data_type(data_type),
          metric_type(meteic_type),
          view_data(view),
          ntotal(0) {
        if (data_type == DataFormatEnum::fp32) {
            code_size = sizeof(fp32) * d;
        } else if (data_type == DataFormatEnum::fp16) {
            code_size = sizeof(fp16) * d;
        } else if (data_type == DataFormatEnum::bf16) {
            code_size = sizeof(bf16) * d;
        } else {
            FAISS_THROW_MSG("OffsetIndex only support float data type.");
        }
    }

    virtual ~DataViewIndexBase() {};

    virtual void
    Train(idx_t n, const void* x) = 0;

    virtual void
    Add(idx_t n, const void* x, const float* norms) = 0;

    virtual void
    Search(const idx_t n, const void* x, const idx_t k, float* distances, idx_t* labels, const BitsetView& bitset) const = 0;
    
    // knn search on selected_ids, selected_ids_lims
    virtual void
    SearchWithIds(const idx_t n, const void* x, const idx_t* ids_num_lims, const idx_t* ids, const idx_t k, float* out_dist, idx_t* out_ids)  const = 0;

    virtual RangeSearchResult
    RangeSearch(const idx_t n, const void* x, const float radius, const float range_filter, const BitsetView& bitset) const = 0;

    virtual RangeSearchResult
    RangeSearchWithIds(const idx_t n, const void* x, const idx_t* ids_num_lims, const idx_t* ids, const float radius, const float range_filter) const = 0;

    virtual void
    ComputeDistanceSubset(const void* x, const idx_t sub_y_n, float* x_y_distances, const idx_t* x_y_labels)  const = 0;
};

struct DataViewIndexFlat : DataViewIndexBase {
    //std::shared_ptr<ThreadPool> search_pool_;
    void
    Train(idx_t n, const void* x) override {
        // do nothing
        return;
    }
    
    void
    Add(idx_t n, const void* x, const float* in_norms) override {
        if (is_cosine) {
            if (in_norms == nullptr) {
                std::vector<float> l2_norms;
                if (data_type == DataFormatEnum::fp32) {
                    l2_norms = GetL2Norms<fp32>((const fp32*)x, d, n);
                } else if (data_type == DataFormatEnum::fp16) {
                    l2_norms = GetL2Norms<fp16>((const fp16*)x, d, n);
                } else {
                    l2_norms = GetL2Norms<bf16>((const bf16*)x, d, n);
                }
                std::unique_lock lock(norms_mutex_);
                norms.insert(norms.end(), l2_norms.begin(), l2_norms.end());
            } else {
                std::unique_lock lock(norms_mutex_);
                norms.insert(norms.end(), in_norms, in_norms + n);
            }
        }
         ntotal.fetch_add(n);
    }

    void
     Search(const idx_t n, const void* x, const idx_t k, float* distances, idx_t* labels, const BitsetView& bitset) const override;

    void
    SearchWithIds(const idx_t n, const void* x, const idx_t* ids_num_lims, const idx_t* ids, const idx_t k, float* out_dist, idx_t* out_ids)  const override;
    
    RangeSearchResult
     RangeSearch(const idx_t n, const void* x, const float radius, const float range_filter, const BitsetView& bitset) const override;

    RangeSearchResult
    RangeSearchWithIds(const idx_t n, const void* x, const idx_t* ids_num_lims, const idx_t* ids, const float radius, const float range_filter) const override;

    void
    ComputeDistanceSubset(const void* x, const idx_t sub_y_n, float* x_y_distances, const idx_t* x_y_labels) const override;
};


template <typename DataType, typename Distance1, typename Distance4, bool NeedNormalize = false>
struct DataViewDistanceComputer : faiss::DistanceComputer {
    ViewDataOp view_data;
    size_t dim;
    const DataType* q;
    Distance1 dist1;
    Distance4 dist4;
    float q_norm;

    DataViewDistanceComputer(const DataViewIndexBase* index, Distance1 dist1, Distance4 dist4,
                             const DataType* query = nullptr)
        : view_data(index->view_data),
          dim(index->d),
          dist1(dist1),
          dist4(dist4) {
        return;
    }

    void
    set_query(const float* x) override {
        q = (const DataType*)x;
        if constexpr (NeedNormalize) {
            q_norm = GetL2Norm(q, dim);
        } 
    }

    float
    operator()(idx_t i) override {
        auto code = view_data(i);
        return distance_to_code(code);
    }

    float
    distance_to_code(const void* x) {
        if constexpr (NeedNormalize) {
            return dist1(q, (const DataType*)x, dim)/q_norm; 
        } else {
            return dist1(q, (const DataType*)x, dim);
        }
    }

    void
    distances_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3, float& dis0, float& dis1,
                      float& dis2, float& dis3) override {
        auto x0 = (DataType*)view_data(idx0);
        auto x1 = (DataType*)view_data(idx1);
        auto x2 = (DataType*)view_data(idx2);
        auto x3 = (DataType*)view_data(idx3);
        dist4(q, x0, x1, x2, x3, dim, dis0, dis1, dis2, dis3);
        if constexpr (NeedNormalize) {
            dis0 /= q_norm;
            dis1 /= q_norm;
            dis2 /= q_norm;
            dis3 /= q_norm;
        }
    }

    /// compute distance between two stored vectors
    float
    symmetric_dis(idx_t i, idx_t j) override {
        auto x = (DataType*)view_data(i);
        auto y = (DataType*)view_data(j);
        return dist1(x, y, dim);
    }
};

std::unique_ptr<faiss::DistanceComputer>
select_data_view_computer(const DataViewIndexBase* index) {
    if (index->data_type == DataFormatEnum::fp16) {
        if (index->metric_type == metric::IP) {
            if (index->is_cosine) {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<fp16, decltype(faiss::fp16_vec_inner_product),
                                                decltype(faiss::fp16_vec_inner_product_batch_4), true>(
                        index, faiss::fp16_vec_inner_product, faiss::fp16_vec_inner_product_batch_4));
            } else {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<fp16, decltype(faiss::fp16_vec_inner_product),
                                                decltype(faiss::fp16_vec_inner_product_batch_4), false>(
                        index, faiss::fp16_vec_inner_product, faiss::fp16_vec_inner_product_batch_4));
            }
        } else{
            return std::unique_ptr<faiss::DistanceComputer>(
                new DataViewDistanceComputer<fp16, decltype(faiss::fp16_vec_L2sqr),
                                             decltype(faiss::fp16_vec_L2sqr_batch_4)>(index, faiss::fp16_vec_L2sqr,
                                                                                      faiss::fp16_vec_L2sqr_batch_4));
        }
    } else if (index->data_type == DataFormatEnum::bf16) {
        if (index->metric_type == metric::IP) {
            if (index->is_cosine) {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<bf16, decltype(faiss::bf16_vec_inner_product),
                                                decltype(faiss::bf16_vec_inner_product_batch_4), true>(
                        index, faiss::bf16_vec_inner_product, faiss::bf16_vec_inner_product_batch_4));
            } else {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<bf16, decltype(faiss::bf16_vec_inner_product),
                                                decltype(faiss::bf16_vec_inner_product_batch_4), false>(
                        index, faiss::bf16_vec_inner_product, faiss::bf16_vec_inner_product_batch_4));
            }
        } else {
            return std::unique_ptr<faiss::DistanceComputer>(
                new DataViewDistanceComputer<bf16, decltype(faiss::bf16_vec_L2sqr),
                                             decltype(faiss::bf16_vec_L2sqr_batch_4)>(index, faiss::bf16_vec_L2sqr,
                                                                                      faiss::bf16_vec_L2sqr_batch_4));
        }
    } else if (index->data_type == DataFormatEnum::fp32) {
        if (index->metric_type == metric::IP) {
            if (index->is_cosine) {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<fp32, decltype(faiss::fvec_inner_product),
                                                decltype(faiss::fvec_inner_product_batch_4), true>(
                        index, faiss::fvec_inner_product, faiss::fvec_inner_product_batch_4));
            } else {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<fp32, decltype(faiss::fvec_inner_product),
                                                decltype(faiss::fvec_inner_product_batch_4), false>(
                        index, faiss::fvec_inner_product, faiss::fvec_inner_product_batch_4));
            }
        } else {
            return std::unique_ptr<faiss::DistanceComputer>(
                new DataViewDistanceComputer<fp32, decltype(faiss::fvec_L2sqr),
                                             decltype(faiss::fvec_L2sqr_batch_4)>(index, faiss::fvec_L2sqr,
                                                                                  faiss::fvec_L2sqr_batch_4));
        }
    } else {
        return nullptr;
    }
}

template <class SingleResultHandler, class SelectorHelper>
void exhaustive_seq_impl(
        const std::unique_ptr<faiss::DistanceComputer>& computer,
        size_t ny,
        SingleResultHandler& resi,
        const SelectorHelper& selector,
        const float* norms = nullptr) { 
    auto filter = [&selector](const size_t j) {
        return selector.is_member(j);
    };
    if (norms != nullptr) {
        auto apply = [&resi, &norms](const float dis, const idx_t j) {
            auto dist_with_norm = dis / (norms[j]);
            resi.add_result(dis, j);
        };
        faiss::distance_compute_if(ny, computer.get(), filter, apply);
    } else {
        auto apply = [&resi](const float dis, const idx_t j) {
            resi.add_result(dis, j);
        };
        faiss::distance_compute_if(ny, computer.get(), filter, apply); 
    }
}

void DataViewIndexFlat::Search(const idx_t n, const void* x, const idx_t k, float* distances, idx_t* labels, const BitsetView& bitset)const {
    auto computer = select_data_view_computer(this);
    std::shared_ptr<float[]> base_norms = nullptr;
    if (is_cosine) {
        std::shared_lock lock(norms_mutex_);
        base_norms = std::shared_ptr<float[]>(new float[norms.size()]);
        std::memcpy(base_norms.get(), norms.data(), sizeof(float) * norms.size());
    }
    if (k < faiss::distance_compute_min_k_reservoir) {
        if (metric_type == metric::L2) { 
            faiss::HeapBlockResultHandler<CMAX> res(n, distances , labels, k);
            faiss::HeapBlockResultHandler<CMAX>::SingleResultHandler resi(res);
            for (auto i = 0; i < n; i++) {
                computer->set_query((const float*)(x + code_size * i));
                resi.begin(i);
                if (bitset.empty()) {
                    exhaustive_seq_impl(computer, n, resi, faiss::IDSelectorAll(), base_norms.get());
                } else {
                    exhaustive_seq_impl(computer, n, resi, BitsetViewIDSelector(bitset), base_norms.get());
                }
                resi.end();
            }
        } else {
            faiss::HeapBlockResultHandler<CMIN> res(n, distances , labels, k);
            faiss::HeapBlockResultHandler<CMIN>::SingleResultHandler resi(res);
            for (auto i = 0; i < n; i++) {
                computer->set_query((const float*)(x + code_size * i));
                resi.begin(i);
                if (bitset.empty()) {
                    exhaustive_seq_impl(computer, n, resi, faiss::IDSelectorAll(), base_norms.get());
                } else {
                    exhaustive_seq_impl(computer, n, resi, BitsetViewIDSelector(bitset), base_norms.get());
                }
                resi.end();
            }
        }
    } else {
        if (metric_type == metric::L2) { 
            faiss::ReservoirBlockResultHandler<CMAX> res(n, distances, labels, k);
            faiss::ReservoirBlockResultHandler<CMAX>::SingleResultHandler resi(res);
            for (auto i = 0; i < n; i++) {
                computer->set_query((const float*)(x + code_size * i));
                resi.begin(i);
                if (bitset.empty()) {
                    exhaustive_seq_impl(computer, n, resi, faiss::IDSelectorAll(), base_norms.get());
                } else {
                    exhaustive_seq_impl(computer, n, resi, BitsetViewIDSelector(bitset), base_norms.get());
                }
                resi.end();
            }
        } else {
            faiss::ReservoirBlockResultHandler<CMIN> res(n, distances, labels, k); 
            faiss::ReservoirBlockResultHandler<CMIN>::SingleResultHandler resi(res);
            for (auto i = 0; i < n; i++) {
                computer->set_query((const float*)(x + code_size * i));
                resi.begin(i);
                if (bitset.empty()) {
                    exhaustive_seq_impl(computer, n, resi, faiss::IDSelectorAll(), base_norms.get());
                } else {
                    exhaustive_seq_impl(computer, n, resi, BitsetViewIDSelector(bitset), base_norms.get());
                }
                resi.end();
            }
        }
    }
}

void
DataViewIndexFlat::SearchWithIds(const idx_t n, const void* x, const idx_t* ids_num_lims, const idx_t* ids, const idx_t k, float* out_dist, idx_t* out_ids)  const {
    for (auto i = 0; i < n; i++) {
        auto base_ids = ids + ids_num_lims[i];
        auto base_n = ids_num_lims[i+1] - ids_num_lims[i];
        auto base_dist = std::unique_ptr<float []>(new float [base_n]);
        auto x_i = x + code_size * i;
        FAISS_THROW_IF_NOT(base_n >= k);
        ComputeDistanceSubset(x_i, base_n, base_dist.get(), base_ids);
        if (is_cosine) {
            std::shared_lock lock(norms_mutex_);
            for (auto j = 0; j < base_n; j++) {
                base_dist[j] = base_dist[j]/norms[base_ids[j]];
            }
        }
        if (metric_type == metric::L2) {
            faiss::reorder_2_heaps<idx_t, CMAX>(1, k, out_ids, out_dist, base_n, base_ids, base_dist.get());
        } else {
            faiss::reorder_2_heaps<idx_t, CMIN>(1, k, out_ids, out_dist, base_n, base_ids, base_dist.get());
        }
    }
}

RangeSearchResult DataViewIndexFlat::RangeSearch(const idx_t n, const void* x, const float radius, const float range_filter, const BitsetView& bitset) const {
    std::vector<std::vector<float>> result_dist_array(n);
    std::vector<std::vector<idx_t>> result_id_array(n);
    auto computer = select_data_view_computer(this);
    std::shared_ptr<float[]> base_norms = nullptr;
    if (is_cosine) {
        std::shared_lock lock(norms_mutex_);
        base_norms = std::shared_ptr<float[]>(new float[norms.size()]);
        std::memcpy(base_norms.get(), norms.data(), sizeof(float) * norms.size());
    }
    auto is_ip = metric_type == metric::IP;
    if (metric_type == metric::L2) {
        faiss::RangeSearchResult res(1);
        faiss::RangeSearchBlockResultHandler<CMAX> resh(&res, radius);
        faiss::RangeSearchBlockResultHandler<CMAX>::SingleResultHandler reshi(resh);
        for (auto i = 0; i < n; i++) {
            computer->set_query(((const float*)x + code_size * i));
            reshi.begin(i);
            if (bitset.empty()) {
                exhaustive_seq_impl(computer, n, reshi, faiss::IDSelectorAll(), base_norms.get());
            } else {
                exhaustive_seq_impl(computer, n, reshi, BitsetViewIDSelector(bitset), base_norms.get());
            }
            reshi.end();
            auto elem_cnt = res.lims[1];
            result_dist_array[i].resize(elem_cnt);
            result_id_array[i].resize(elem_cnt);
            for (size_t j = 0; j < elem_cnt; j++) {
                result_dist_array[i][j] = res.distances[j];
                result_id_array[i][j] = res.labels[j];
            }
            if (range_filter != defaultRangeFilter) {
                FilterRangeSearchResultForOneNq(result_dist_array[i], result_id_array[i], is_ip, radius,
                                                range_filter);
            }
        }
    } else {
        faiss::RangeSearchResult res(1);
        faiss::RangeSearchBlockResultHandler<CMIN> resh(&res, radius);
        faiss::RangeSearchBlockResultHandler<CMIN>::SingleResultHandler reshi(resh);
        for (auto i = 0; i < n; i++) {
            computer->set_query(((const float*)x + code_size * i));
            reshi.begin(i);
            if (bitset.empty()) {
                exhaustive_seq_impl(computer, n, reshi, faiss::IDSelectorAll(), base_norms.get());
            } else {
                exhaustive_seq_impl(computer, n, reshi, BitsetViewIDSelector(bitset), base_norms.get());
            }
            reshi.end();
            auto elem_cnt = res.lims[1];
            result_dist_array[i].resize(elem_cnt);
            result_id_array[i].resize(elem_cnt);
            for (size_t j = 0; j < elem_cnt; j++) {
                result_dist_array[i][j] = res.distances[j];
                result_id_array[i][j] = res.labels[j];
            }
            if (range_filter != defaultRangeFilter) {
                FilterRangeSearchResultForOneNq(result_dist_array[i], result_id_array[i], is_ip, radius,
                                                range_filter);
            }
        }
    }
    return GetRangeSearchResult(result_dist_array, result_id_array, is_ip, n, radius, range_filter);
}

RangeSearchResult
DataViewIndexFlat::RangeSearchWithIds(const idx_t n, const void* x, const idx_t* ids_num_lims, const idx_t* ids, const float radius, const float range_filter) const {
    std::vector<std::vector<float>> result_dist_array(n);
    std::vector<std::vector<idx_t>> result_id_array(n);
    auto is_ip = metric_type == metric::IP;
    for (auto i = 0; i < n; i++) {
        auto base_ids = ids + ids_num_lims[i];
        auto base_n = ids_num_lims[i+1] - ids_num_lims[i];
        auto base_dist = std::unique_ptr<float []>(new float[base_n]);
        auto x_i = x + code_size * i;
        ComputeDistanceSubset(x_i, base_n, base_dist.get(), base_ids);
        if (is_cosine) {
            std::shared_lock lock(norms_mutex_);
            for (auto j = 0; j < base_n; j++) {
                base_dist[j] = base_dist[j]/norms[base_ids[j]];
            }
        }
        for (auto j = 0; j < base_n; j++) {
            if (!is_ip) {
                if (base_dist[j] < radius) {
                    result_dist_array[i].emplace_back(base_dist[j]);
                    result_id_array[i].emplace_back(base_ids[j]);
                }
            } else {
                if (base_dist[j] > radius) {
                    result_dist_array[i].emplace_back(base_dist[j]);
                    result_id_array[i].emplace_back(base_ids[j]);
                }
            } 
        }
        if (range_filter != defaultRangeFilter) {
            FilterRangeSearchResultForOneNq(result_dist_array[i], result_id_array[i], is_ip, radius,
                                            range_filter);
        }
    }
    return GetRangeSearchResult(result_dist_array, result_id_array, is_ip, n, radius, range_filter);
}

void DataViewIndexFlat::ComputeDistanceSubset(const void* x, const idx_t sub_y_n, float* x_y_distances, const idx_t* x_y_labels) const {
    auto computer = select_data_view_computer(this);
    computer->set_query((const float*)(x));
    const int64_t* __restrict idsj = x_y_labels;
    float* __restrict disj = x_y_distances;
    
    auto filter = [=](const size_t i) { return (idsj[i] >= 0); };
    auto apply = [=](const float dis, const size_t i) {
        disj[i] = dis;
    };
    distance_compute_by_idx_if(idsj, sub_y_n, computer.get(), filter, apply);
}
}  // namespace knowhere