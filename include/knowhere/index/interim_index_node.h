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

#ifndef INTERIM_INDEX_NODE_H 
#define INTERIM_INDEX_NODE_H
#include "folly/concurrency/AtomicSharedPtr.h"
#include "knowhere/index/index_node.h"
#include "knowhere/utils.h"
#include "knowhere/comp/thread_safe_state_controller.h"
#include "knowhere/comp/brute_force.h"
namespace knowhere {
template <typename IndexType, typename DataType>
class InterimIndexNode : public IndexNode {
 public:
    template <typename... Args>
    InterimIndexNode(Args&&... args) {
        index_node_ = std::make_unique<IndexType>(IndexType(std::forward<Args>(args)...));
    }

    ~InterimIndexNode() {
        async_ctl_->cancel();
        index_node_ = nullptr;
        raw_data_ = nullptr;
    }

    Status
    Train(const DataSetPtr dataset, const Config& cfg);
    Status
    Add(const DataSetPtr dataset, const Config& cfg);

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const override;

    expected<std::vector<IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const override;
    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override;
    bool
    HasRawData(const std::string& metric_type) const override;

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override;
    Status
    Serialize(BinarySet& binset) const override;

    Status
    Deserialize(const BinarySet& binset, const Config& config) override;
    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override;

    std::unique_ptr<BaseConfig>
    CreateConfig() const override;

    int64_t
    Dim() const override;

    int64_t
    Size() const override;

    int64_t
    Count() const override;

    std::string
    Type() const override;
 private:
    inline Status AsyncBuild(const DataSetPtr& dataset, const Config& cfg, const ThreadSafeStateControllerPtr& async_ctl);
    DataSetPtr raw_data_ = nullptr;
    mutable std::shared_mutex raw_data_mtx_;
    std::unique_ptr<IndexNode> index_node_;
    std::atomic<bool> index_is_ready_ = false;
    mutable std::shared_mutex index_mtx_;
    ThreadSafeStateControllerPtr async_ctl_ = std::make_shared<ThreadSafeStateController>(); 
};

}  // namespace knowhere

#endif /* INTERIM_INDEX_NODE_H */
