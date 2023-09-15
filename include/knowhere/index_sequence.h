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

#ifndef INDEX_SEQUENCE_H
#define INDEX_SEQUENCE_H
#include <iostream>
#include <memory>
namespace knowhere {
class IndexSequence {
 public:
    IndexSequence(std::unique_ptr<uint8_t[]>&& seq = nullptr, size_t seq_size = 0) {
        this->seq = std::move(seq);
        this->size = seq_size;
    }

    IndexSequence(IndexSequence&& index_seq) {
        *this = std::move(index_seq);
    }

    IndexSequence(const IndexSequence&) = delete;

    IndexSequence&
    operator=(IndexSequence&& index_seq) {
        seq = std::move(index_seq.seq);
        size = index_seq.size;
        index_seq.size = 0;
        return *this;
    }

    IndexSequence&
    operator=(const IndexSequence&) = delete;

    size_t
    GetSize() const {
        return size;
    }

    uint8_t*
    GetSeq() const {
        return seq.get();
    }

    std::unique_ptr<uint8_t[]>
    StealSeq() {
        size = 0;
        return std::move(seq);
    };

    bool
    Empty() const {
        return (seq == nullptr) || (size == 0);
    }

 private:
    std::unique_ptr<uint8_t[]> seq = nullptr;
    size_t size = 0;
};
}  // namespace knowhere
#endif /* INDEX_SEQUENCE_H */
