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
#ifndef INDEX_NODE_WITH_DATA_VIEW_REFINER_H
#define INDEX_NODE_WITH_DATA_VIEW_REFINER_H
#include <atomic>
#include "knowhere/index/index_node.h"
#include "index/data_view_dense_index/data_view_dense_index.h"
#include "index/data_view_dense_index/data_view_index_node_config.h"
#include <random>
namespace knowhere {
template <typename DataType, const char* BaseIndexName>
class IndexNodeWithDataViewRefiner : public IndexNode {
    // quant index + refine distance computer mode
    static_assert(KnowhereFloatTypeCheck<DataType>::value);
   // static_assert(std::is_same_v<IndexType, IvfIndexNode<fp32, faiss::IndexScaNN>>, "not support");

 public:
    IndexNodeWithDataViewRefiner(const int32_t& version, const Object& object): {
        auto data_view_index_pack = dynamic_cast<const Pack<ViewDataOp>*>(&object);
        assert(data_view_index_pack != nullptr);
        view_data_op_ = data_view_index_pack->GetPack();
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg) override;

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg) override;

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override;

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "Offset Index not maintain raw data.");
    }

    static Status
    StaticConfigCheck(const Config& cfg, PARAM_TYPE paramType, std::string& msg) {
        auto base_cfg = static_cast<const BaseConfig&>(cfg);
        if constexpr (KnowhereFloatTypeCheck<DataType>::value) {
            if (IsMetricType(base_cfg.metric_type.value(), metric::L2) ||
                IsMetricType(base_cfg.metric_type.value(), metric::IP) ||
                IsMetricType(base_cfg.metric_type.value(), metric::COSINE)) {
            } else {
                msg = "metric type " + base_cfg.metric_type.value() +
                      " not found or not supported, supported: [L2 IP COSINE]";
                return Status::invalid_metric_type;
            }
        }
        return Status::success;
    }

    static bool
    CommonHasRawData() {
        return false;
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        return false;
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_ERROR_ << "OffsetIndex is a JIT index, should not serialize";
        return Status::not_implemented;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override {
        LOG_KNOWHERE_ERROR_ << "OffsetIndex is a JIT index, should not deserialize";
        return Status::not_implemented;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> cfg) override {
        LOG_KNOWHERE_ERROR_ << "OffsetIndex is a JIT index, should not deserializefromfile";
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        if (std::string(BaseIndexName) == IndexEnum::INDEX_FAISS_SCANN) {
            return std::make_unique<IndexSCANNWithDataViewReinferConfig>();
        } else {
            return std::make_unique<IndexWithDataViewRefinerConfig>();
        }
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    int64_t
    Dim() const override {
        if (!this->base_index_) {
            return -1;
        }
        return this->base_index_->Dim();
    }

    int64_t
    Size() const override {
        if (!this->base_index_) {
            return this->base_index_->Size();
        }
        return 0;
    }

    int64_t
    Count() const override {
        return 0;
    }

    std::string
    Type() const override;
 
 private:
    class iterator : public IndexIterator {
        public:
        iterator(const DataViewIndexBase* refine_offset_index, std::shared_ptr<IndexIterator> quant_workspace,  std::unique_ptr<DataType[]>&& copied_query,  bool larger_is_closer, const float knn_refine_ratio = 0.5f) : 
                  IndexIterator(larger_is_closer, knn_refine_ratio),
                  copied_query_(std::move(copied_query)),
                  quant_workspace_(quant_workspace) {
            refine_computer_ = select_data_view_computer(dynamic_cast<const DataViewIndexBase*>(refine_offset_index));
            refine_computer_->set_query((const float*)copied_query_.get());
        }

        protected:
        void
        next_batch(std::function<void(const std::vector<DistId>&)> batch_handler) override {
            if (quant_workspace_ == nullptr) {
                throw std::runtime_error("quant workspace is null in offset refine index.");
            }
            //quant_workspace_->next_batch(batch_handler);
        }
        float
        raw_distance(int64_t id) override {
            if (refine_computer_ == nullptr) {
                throw std::runtime_error("refine computer is null in offset refine index.");
            }
            return refine_computer_->operator()(id);
        }
        private:
        std::shared_ptr<IndexIterator> quant_workspace_ = nullptr;
        std::unique_ptr<faiss::DistanceComputer> refine_computer_ = nullptr;
        std::unique_ptr<DataType[]> copied_query_ = nullptr;
    };
    bool is_cosine_;  
    ViewDataOp view_data_op_; 
    std::unique_ptr<DataViewIndexFlat> refine_offset_index_; // a offset flat index to maintain raw data without extra memory
    std::unique_ptr<IndexNode> base_index_;  // quant_index will hold codes in memory, datatype is fp32
};

namespace {
constexpr int64_t kBatchSize = 4096;
constexpr int64_t kMaxTrainSize = 5000;
constexpr const char* kIndexNodeSuffixWithDataViewRefiner = "_with_data_refiner";
void 
MatchQuantIndexConfig(Config* cfg) {
    if (cfg == nullptr) return;
    if (auto scann_cfg = dynamic_cast<ScannConfig*>(cfg)) {
        scann_cfg->with_raw_data = false;
    } 
    auto base_cfg = dynamic_cast<BaseConfig*>(cfg);
    if (base_cfg->metric_type == metric::COSINE) {
        base_cfg->metric_type = metric::IP;
    }
}
template <typename DataType>
inline DataSetPtr
GenFp32TrainDataSet(const DataSetPtr& src, bool is_cosine = false) {
    DataSetPtr train_ds;
    auto rows = src->GetRows();
    bool need_copy = false;
    if (rows <= kMaxTrainSize) {
        train_ds = ConvertFromDataTypeIfNeeded<DataType>(src);
        if constexpr (std::is_same_v<DataType, fp32>) {
            need_copy = true;
        } 
    } else {
        std::vector<int64_t> random_ids(rows);
        std::iota(random_ids.begin(), random_ids.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(random_ids.begin(), random_ids.end(), gen);
        auto dim = src->GetDim();
        const DataType* src_data = (const DataType*)src->GetTensor();
        auto* des_data = new float[dim * kMaxTrainSize];
        for (auto i = 0; i < kMaxTrainSize; i++) {
            auto from_id = random_ids[i];
            auto to_id = i;
            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                std::memcpy(des_data + to_id * dim, src_data + from_id * dim, sizeof(float) * dim);
            } else {
                for (auto d = 0; d < dim; d++) {
                    // optimize it with simd
                    des_data[to_id * dim + d] = (fp32)src_data[from_id * dim + d];
                }
            }
        }
        auto des = std::make_shared<DataSet>();
        des->SetRows(kMaxTrainSize);
        des->SetDim(dim);
        des->SetTensor(des_data);
        des->SetIsOwner(true);
        train_ds = des;
    }
    if (is_cosine) {
        if (need_copy) {
            train_ds = std::get<0>(CopyAndNormalizeDataset<fp32>(train_ds));
        } else {
            NormalizeDataset<fp32>(train_ds);
        }
    }
    return train_ds;
}  
template <typename DataType>
inline DataSetPtr
ConvertToFp32DataSet(const DataSetPtr& src, bool is_cosine = false, const std::optional<int64_t> start = std::nullopt,
                            const std::optional<int64_t> count = std::nullopt) {
    auto fp32_ds = ConvertFromDataTypeIfNeeded<DataType>(src, start, count);
     if (is_cosine) {
        if constexpr (std::is_same_v<DataType, fp32>) {
            fp32_ds = std::get<0>(CopyAndNormalizeDataset<fp32>(fp32_ds));
        } else {
            NormalizeDataset<fp32>(fp32_ds);
        }
    }
    return fp32_ds;
}
} // namespace 

template <typename DataType, const char* BaseIndexName>
Status
IndexNodeWithDataViewRefiner<DataType, BaseIndexName>::Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg) {
    IndexWithDataViewRefinerConfig& base_cfg = static_cast<IndexWithDataViewRefinerConfig&>(*cfg);

    this->is_cosine_ = IsMetricType(base_cfg.metric_type.value(), knowhere::metric::COSINE);
    auto rows = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto data = dataset->GetTensor();
    // construct refine index:
    auto refine_metric = is_cosine_?metric::IP:base_cfg.metric_type.value();
    refine_offset_index_ = std::make_unique<DataViewIndexFlat>(dim, datatype_v<DataType>, refine_metric, this->view_data_op_);
    // construct quant index and train:
    MatchQuantIndexConfig(cfg.get());
    //base_index_ = std::shared_ptr<IndexNode>();
    auto fp32_train_ds = GenFp32TrainDataSet<DataType>(dataset);
    base_index_->Train(fp32_train_ds, cfg);
}

template <typename DataType, const char* BaseIndexName>
Status
IndexNodeWithDataViewRefiner<DataType, BaseIndexName>::Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg) {
    auto rows = dataset->GetRows();
    auto data = dataset->GetTensor();
    MatchQuantIndexConfig(cfg.get());
    for (auto blk_i = 0; blk_i < rows; blk_i += kBatchSize) {
        auto blk_size = std::min(kBatchSize, rows - blk_i);
        auto fp32_ds = ConvertToFp32DataSet<DataType>(dataset, is_cosine_, blk_i, blk_size);
        base_index_->Add(fp32_ds, cfg);
    }
    refine_offset_index_->Add(rows, data, nullptr);
}

template <typename DataType, const char* BaseIndexName>
expected<DataSetPtr>
IndexNodeWithDataViewRefiner<DataType, BaseIndexName>::Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const {
    if (this->base_index_ == nullptr|| this->refine_offset_index_ == nullptr) {
        LOG_KNOWHERE_WARNING_ << "search on empty index";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not is trained.");
    }  
    MatchQuantIndexConfig(cfg.get());
    IndexWithDataViewRefinerConfig& base_cfg = static_cast<IndexWithDataViewRefinerConfig&>(*cfg);
    auto nq = dataset->GetRows();
    auto topk = base_cfg.k.value();
    auto refine_k = int(topk * base_cfg.knn_refine_ratio.value());
    base_cfg.k = refine_k;
    auto fp32_ds = ConvertToFp32DataSet<DataType>(dataset, is_cosine_);  
    auto quant_res = Search(fp32_ds, std::move(cfg), bitset);
    if (!quant_res.has_value()) {
        return quant_res;
    }
    auto refine_ids = quant_res.value()->GetIds();
    auto labels = std::make_unique<int64_t[]>(nq * topk);
    auto distances = std::make_unique<float[]>(nq * topk);
    std::fill(distances.get(), distances.get() + nq * topk, std::numeric_limits<float>::quiet_NaN());
    std::fill(labels.get(), labels.get() + nq * topk, -1);
    auto queries_lims =std::vector<faiss::idx_t>(nq + 1);
    for (auto i = 0; i < nq + 1; i++) {
        queries_lims[i] = topk * i;
    }
    refine_offset_index_->SearchWithIds(nq, dataset->GetTensor(), queries_lims.data(), refine_ids, topk, distances.get(), labels.get());
    return GenResultDataSet(nq, topk, std::move(labels), std::move(distances));
}
    
template <typename DataType, const char* BaseIndexName>
expected<DataSetPtr>
IndexNodeWithDataViewRefiner<DataType, BaseIndexName>::RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const {
    if (this->base_index_ == nullptr|| this->refine_offset_index_ == nullptr) {
        LOG_KNOWHERE_WARNING_ << "search on empty index";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not is trained.");
    }  
    MatchQuantIndexConfig(cfg.get());
    const IndexWithDataViewRefinerConfig& base_cfg = static_cast<const IndexWithDataViewRefinerConfig&>(*cfg);
    auto nq = dataset->GetRows();
    auto radius = base_cfg.radius.value();
    auto range_filter = base_cfg.range_filter.value();
    auto fp32_ds = ConvertToFp32DataSet<DataType>(dataset, is_cosine_);  
    auto quant_res = base_index_->RangeSearch(fp32_ds, std::move(cfg), bitset);
    if (!quant_res.has_value()) {
        return quant_res;
    }
    auto quant_res_ids = quant_res.value()->GetIds();
    auto quant_res_lims = quant_res.value()->GetLims();
    auto final_res = refine_offset_index_->RangeSearchWithIds(nq, dataset->GetTensor(), (const knowhere::idx_t*)quant_res_lims, (const knowhere::idx_t*)quant_res_ids, radius, range_filter);
    return GenResultDataSet(nq, std::move(final_res));
}

template <typename DataType, const char* BaseIndexName>
expected<std::vector<IndexNode::IteratorPtr>>
IndexNodeWithDataViewRefiner<DataType, BaseIndexName>::AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const {
    if (this->base_index_ == nullptr|| this->refine_offset_index_ == nullptr) {
        LOG_KNOWHERE_WARNING_ << "search on empty index";
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::empty_index, "index not is trained.");
    } 
    
    auto dim = dataset->GetDim();
    auto rows = dataset->GetRows();
    auto data = dataset->GetTensor();
    MatchQuantIndexConfig(cfg.get());
    const auto& base_cfg = static_cast<const BaseConfig&>(*cfg);
    auto knn_refine_ratio = base_cfg.knn_refine_ratio.value(); 
    auto larger_is_closer = IsMetricType(base_cfg.metric_type.value(), knowhere::metric::IP) || is_cosine_;

    auto fp32_ds = ConvertToFp32DataSet<DataType>(dataset, is_cosine_);  
    auto quant_init = AnnIterator(fp32_ds, std::move(cfg),bitset);
    if (!quant_init.has_value()) {
        return quant_init;
    }
    auto quant_workspace_vec = quant_init.value();
    if (quant_workspace_vec.size() != rows) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::internal_error, "quant workspace is not equal to the rows count of input dataset.");
    }
    auto vec = std::vector<IndexNode::IteratorPtr>(rows, nullptr);
    for (auto i = 0; i < rows; i++) {
        auto cur_query = (const DataType*)data + i * dim;
        std::unique_ptr<DataType[]> copied_query = nullptr;
        if (is_cosine_) {
            copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
        } else {
            copied_query = std::make_unique<DataType[]>(dim);
            std::copy_n(cur_query, dim, copied_query.get());
        }
        //iterator(std::shared_ptr<IndexIterator> quant_workspace,  std::unique_ptr<void*>&& copied_query,  bool larger_is_closer, const float knn_refine_ratio = 0.5f) : 
        vec[i] = std::make_shared<iterator>(new iterator(this->refine_offset_index_.get(), std::dynamic_pointer_cast<IndexIterator>(quant_workspace_vec[i]), std::move(copied_query), larger_is_closer, knn_refine_ratio));
    }
    return vec;
}

template <typename DataType, const char* BaseIndexName>
std::string
IndexNodeWithDataViewRefiner<DataType, BaseIndexName>::Type() const {
    return std::string(BaseIndexName) + kIndexNodeSuffixWithDataViewRefiner;
}

}  // namespace knowhere
#endif