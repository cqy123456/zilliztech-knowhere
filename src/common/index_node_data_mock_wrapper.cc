// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "knowhere/index_node_data_mock_wrapper.h"

#include "knowhere/comp/thread_pool.h"
#include "knowhere/index_node.h"
#include "knowhere/utils.h"

namespace knowhere {

template <typename DataType>
Status
IndexNodeDataMockWrapper<DataType>::Build(const DataSet& dataset, const Config& cfg) {
    std::shared_ptr<const DataSet> ds_ptr = nullptr;
    if (this->Type() != knowhere::IndexEnum::INDEX_DISKANN) {
        ds_ptr = dataset.Get();
        if constexpr (!std::is_same_v<DataType, typename MockData<DataType>::type>) {
            ds_ptr = data_type_conversion<DataType, typename MockData<DataType>::type>(dataset);
        }
    }
    return index_node_->Build(*ds_ptr, cfg);
}

template <typename DataType>
Status
IndexNodeDataMockWrapper<DataType>::Train(const DataSet& dataset, const Config& cfg) {
    std::shared_ptr<const DataSet> ds_ptr = nullptr;
    ds_ptr = dataset.Get();
    if constexpr (!std::is_same_v<DataType, typename MockData<DataType>::type>) {
        ds_ptr = data_type_conversion<DataType, typename MockData<DataType>::type>(dataset);
    }
    return index_node_->Train(*ds_ptr, cfg);
}

template <typename DataType>
Status
IndexNodeDataMockWrapper<DataType>::Add(const DataSet& dataset, const Config& cfg) {
    std::shared_ptr<const DataSet> ds_ptr = nullptr;
    ds_ptr = dataset.Get();
    if constexpr (!std::is_same_v<DataType, typename MockData<DataType>::type>) {
        ds_ptr = data_type_conversion<DataType, typename MockData<DataType>::type>(dataset);
    }
    return index_node_->Add(*ds_ptr, cfg);
}

template <typename DataType>
expected<DataSetPtr>
IndexNodeDataMockWrapper<DataType>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    auto ds_ptr = dataset.Get();
    if constexpr (!std::is_same_v<DataType, typename MockData<DataType>::type>) {
        ds_ptr = data_type_conversion<DataType, typename MockData<DataType>::type>(dataset);
    }
    return index_node_->Search(*ds_ptr, cfg, bitset);
}

template <typename DataType>
expected<DataSetPtr>
IndexNodeDataMockWrapper<DataType>::RangeSearch(const DataSet& dataset, const Config& cfg,
                                                const BitsetView& bitset) const {
    auto ds_ptr = dataset.Get();
    if constexpr (!std::is_same_v<DataType, typename MockData<DataType>::type>) {
        ds_ptr = data_type_conversion<DataType, typename MockData<DataType>::type>(dataset);
    }
    return index_node_->RangeSearch(*ds_ptr, cfg, bitset);
}

template <typename DataType>
expected<std::vector<std::shared_ptr<typename IndexNode::iterator>>>
IndexNodeDataMockWrapper<DataType>::AnnIterator(const DataSet& dataset, const Config& cfg,
                                                const BitsetView& bitset) const {
    auto ds_ptr = dataset.Get();
    if constexpr (!std::is_same_v<DataType, typename MockData<DataType>::type>) {
        ds_ptr = data_type_conversion<DataType, typename MockData<DataType>::type>(dataset);
    }
    return index_node_->AnnIterator(*ds_ptr, cfg, bitset);
}

template <typename DataType>
expected<DataSetPtr>
IndexNodeDataMockWrapper<DataType>::GetVectorByIds(const DataSet& dataset) const {
    auto res = index_node_->GetVectorByIds(dataset);
    if constexpr (!std::is_same_v<DataType, typename MockData<DataType>::type>) {
        if (res.has_value()) {
            auto res_v = data_type_conversion<DataType, typename MockData<DataType>::type>(*res.value());
            return res_v;
        } else {
            return res;
        }
    } else {
        return res;
    }
}

template class knowhere::IndexNodeDataMockWrapper<knowhere::fp16>;
template class knowhere::IndexNodeDataMockWrapper<knowhere::bf16>;
}  // namespace knowhere
