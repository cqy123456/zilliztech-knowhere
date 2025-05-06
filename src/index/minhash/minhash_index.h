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

#ifndef MINHASH_TREE_H
#define MINHASH_TREE_H
#include "diskann/utils.h"
#include "faiss/impl/io.h"
#include "io/file_io.h"
#include "io/memory_io.h"
#include "knowhere/comp/bloomfilter.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "simd/hook.h"
namespace knowhere {
using Idx = int64_t;
using KeyType = uint64_t;
using ValueType = Idx;

struct MinHashIndexBuildParams {
    std::string data_path;
    std::string index_file_path;
    size_t band;
    size_t block_size;
    bool has_raw_data;
};

struct MinHashIndexLoadParams {
    std::string index_file_path;
    bool hash_code_in_memory = false;
    bool global_bloom_filter = false;
    float false_positive_prob = 0.01;
};
// struct MinHashIndexSearchParams {
//     size_t k;
//     size_t refine_factor;
// };

struct KVPair {
    KeyType Key;
    ValueType Value;
};
// index of each band
class MinHashBandIndex {
 public:
    static size_t
    FormatAndSave(faiss::BlockFileIOWriter& writer, const KVPair* sorted_kv, const size_t block_size,
                  const size_t rows);

    Status
    Load(FileReader& reader, size_t rows, char* mmap_data, BloomFilter<KeyType>& bloom_filter, bool print);

    std::vector<ValueType>
    Search(KeyType key, bool search_all);

    ~MinHashBandIndex() {
        if (mmap_enable_) {
            munmap(data_, block_size_ * blocks_num_);
        }
    }

 private:
    std::vector<KeyType> mins_;
    std::vector<KeyType> maxs_;
    std::vector<size_t> num_in_a_blk_;
    size_t block_size_;
    size_t blocks_num_;
    bool mmap_enable_ = false;
    char* data_ = nullptr;
    std::unique_ptr<char[]> owned_data_ = nullptr;
};

class MinHashIndexBase {
 public:
    MinHashIndexBase(){};
    virtual Status
    Load(MinHashIndexLoadParams* params) = 0;
    virtual void
    Search(const char* query, float* distances, Idx* labels) = 0;
    virtual void
    BatchSearch(const char* query, size_t nq, float* distances, Idx* labels, std::shared_ptr<ThreadPool> pool) = 0;
    size_t
    Count() {
        return ntotal_;
    };
    size_t
    GetDim() {
        return dim_;
    }
    virtual size_t
    Size() = 0;
    virtual ~MinHashIndexBase() = default;

 protected:
    size_t ntotal_;
    size_t dim_;
};

/* all index meta and codes will maintain as blocks in file*/
// todo: hold raw data for higher recall
template <typename IN_HASH_TYPE>
class MinHashIndex : public MinHashIndexBase {
 public:
    MinHashIndex(){};
    static Status
    BuildAndSave(MinHashIndexBuildParams* params);
    Status
    Load(MinHashIndexLoadParams* params) override;
    void
    Search(const char* query, float* distances, Idx* labels) override;
    void
    BatchSearch(const char* query, size_t nq, float* distances, Idx* labels, std::shared_ptr<ThreadPool> pool) override;
    size_t
    Size() override {
        return this->ntotal_ * band_ * sizeof(KeyType);
    }

    ~MinHashIndex() {
        if (mmap_data_) {
            munmap(mmap_data_, file_size_);
        }
    }

 private:
    std::unique_ptr<MinHashBandIndex[]> band_index_;
    bool is_loaded_ = false;
    size_t block_size_;
    size_t band_;
    char* mmap_data_ = nullptr;
    size_t file_size_;
    bool with_raw_data_ = false;
    IN_HASH_TYPE* raw_data_ = nullptr;  // mmap mode, use IO object later
    std::vector<BloomFilter<KeyType>> bloom_;
};

// todo: thread pool version
namespace {
constexpr int MMAP_IO_FLAGS = MAP_POPULATE | MAP_SHARED;
constexpr int kBatch = 4096;

template <typename T>
inline float
minhash_jaccard(const T* x, const T* y, size_t d, size_t mh_d) {
    // checking d % mh_d == 0 at first
    size_t mh_r = d / mh_d;
    for (size_t i = 0; i < mh_d; i++) {
        const T* x_i = x + mh_r * i;
        const T* y_i = y + mh_r * i;
        size_t j = 0;
        for (; j < mh_r; j++) {
            if (x_i[j] != y_i[j])
                break;
        }
        if (j == mh_r)
            return 1.0;
    }
    return 0.0;
}

// file format:
// n: row count
// size: dim * data_size in bits
template <typename T>
inline void
load_binary_file(const std::string& bin_file, std::unique_ptr<T[]>& data, size_t& npts, size_t& dim) {
    std::ifstream file(bin_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("fail to open file: " + bin_file);
    }
    uint32_t n, d;
    file.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    npts = n;
    dim = d / (8 * sizeof(T));
    uint64_t total_size = dim * npts * sizeof(T);
    data = std::make_unique<T[]>(npts * dim);
    file.read(reinterpret_cast<char*>(data.get()), total_size);
}

template <typename IN_HASH_TYPE>
inline KeyType
get_hash_key(const IN_HASH_TYPE* data, size_t dim, size_t band, size_t band_i) {
    if constexpr (std::is_same_v<IN_HASH_TYPE, uint32_t>) {
        return faiss::calculate_hash((const uint32_t*)data, dim, band, band_i);
    } else {
        auto sub_dim = dim / band;
        auto band_i_data = data + sub_dim * band_i;
        return hash_binary_vec((const uint8_t*)band_i_data, dim * sizeof(IN_HASH_TYPE));
    }
}

template <typename IN_HASH_TYPE>
inline std::shared_ptr<KVPair[]>
gen_transposed_hash_kv(const IN_HASH_TYPE* data, size_t rows, size_t dim, size_t band) {
    auto res_kv = std::shared_ptr<KVPair[]>(new KVPair[band * rows]);
    auto sub_dim = dim / band;
    auto batch_num = (rows + kBatch - 1) / kBatch;
    auto build_pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    for (size_t i = 0; i < batch_num; i++) {
        futures.emplace_back(build_pool->push([&, idx = i]() {
            auto beg_id = idx * kBatch;
            auto end_id = std::min((idx + 1) * kBatch, rows);
            for (size_t j = beg_id; j < end_id; j++) {
                const IN_HASH_TYPE* data_j = data + dim * j;
                for (size_t b = 0; b < band; b++) {
                    KVPair kv = {get_hash_key(data_j, dim, band, b), j};
                    res_kv.get()[b * rows + j] = kv;
                }
            }
        }));
    }
    WaitAllSuccess(futures);
    return res_kv;
}

void
sort_kv(const std::shared_ptr<KVPair[]> kv_code, size_t rows, size_t band) {
    auto build_pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    for (size_t i = 0; i < band; i++) {
        futures.emplace_back(build_pool->push([&, idx = i]() {
            std::sort(kv_code.get() + rows * idx, kv_code.get() + rows * (idx + 1),
                      [](const KVPair& a, const KVPair& b) { return a.Key < b.Key; });
        }));
    }
    WaitAllSuccess(futures);
}
}  // namespace

size_t
MinHashBandIndex::FormatAndSave(faiss::BlockFileIOWriter& writer, const KVPair* sorted_kv, const size_t block_size,
                                const size_t rows) {
    size_t max_num_of_a_block = block_size / sizeof(KVPair);
    size_t blocks_num = (rows + max_num_of_a_block - 1) / max_num_of_a_block;
    std::vector<KeyType> mins;
    std::vector<KeyType> maxs;
    std::vector<size_t> num_in_a_blk;
    mins.resize(blocks_num);
    maxs.resize(blocks_num);
    num_in_a_blk.resize(blocks_num);
    std::unique_ptr<KeyType[]> block_key_buf = std::make_unique<KeyType[]>(max_num_of_a_block);
    std::unique_ptr<ValueType[]> block_val_buf = std::make_unique<ValueType[]>(max_num_of_a_block);
    writer.flush();
    size_t data_pos = writer.tellg();
    for (size_t i = 0; i < blocks_num; i++) {
        writer.flush();
        auto beg = i * max_num_of_a_block;
        auto end = std::min((i + 1) * max_num_of_a_block, rows);
        num_in_a_blk[i] = end - beg;
        mins[i] = sorted_kv[beg].Key;
        maxs[i] = sorted_kv[end - 1].Key;
        for (auto j = 0; j < num_in_a_blk[i]; j++) {
            block_key_buf[j] = sorted_kv[beg + j].Key;
            block_val_buf[j] = sorted_kv[beg + j].Value;
        }
        writer.write((const char*)block_key_buf.get(), num_in_a_blk[i] * sizeof(KeyType));
        writer.write((const char*)block_val_buf.get(), num_in_a_blk[i] * sizeof(ValueType));
    }
    writer.flush();
    auto index_meta_pos = writer.tellg();
    writeBinaryPOD(writer, blocks_num);
    writeBinaryPOD(writer, block_size);
    writeBinaryPOD(writer, data_pos);
    writer.write((const char*)mins.data(), mins.size() * sizeof(KeyType));
    writer.write((const char*)maxs.data(), maxs.size() * sizeof(KeyType));
    writer.write((const char*)num_in_a_blk.data(), num_in_a_blk.size() * sizeof(size_t));
    writer.flush();
    return index_meta_pos;
}

Status
MinHashBandIndex::Load(FileReader& reader, size_t rows, char* mmap_data, BloomFilter<KeyType>& bloom_filter,
                       bool print) {
    size_t data_pos;
    readBinaryPOD(reader, this->blocks_num_);
    readBinaryPOD(reader, this->block_size_);
    readBinaryPOD(reader, data_pos);
    mins_.resize(blocks_num_);
    maxs_.resize(blocks_num_);
    num_in_a_blk_.resize(blocks_num_);
    reader.read((char*)mins_.data(), mins_.size() * sizeof(KeyType));
    reader.read((char*)maxs_.data(), maxs_.size() * sizeof(KeyType));
    reader.read((char*)num_in_a_blk_.data(), num_in_a_blk_.size() * sizeof(size_t));
    if (mmap_data) {
        data_ = mmap_data + data_pos;
    } else {
        owned_data_ = std::make_unique<char[]>(block_size_ * blocks_num_);
        reader.seek(data_pos);
        reader.read(owned_data_.get(), block_size_ * blocks_num_);
        data_ = owned_data_.get();
    }
    auto build_pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    for (auto i = 0; i < blocks_num_; i++) {
        futures.emplace_back(build_pool->push([&, idx = i]() {
            KeyType* blk_i = reinterpret_cast<KeyType*>(data_ + block_size_ * idx);
            for (auto j = 0; j < num_in_a_blk_[idx]; j++) {
                bloom_filter.add(blk_i[j]);
            }
        }));
    }
    WaitAllSuccess(futures);
    return Status::success;
}
std::vector<ValueType>
MinHashBandIndex::Search(KeyType key, bool more_res) {
    auto block_id = faiss::binary_search_ge(maxs_.data(), maxs_.size(), key);
    if (block_id == -1 || key < mins_[block_id]) {
        return {-1};
    }
    auto rows = num_in_a_blk_[block_id];
    KeyType* blk_k = reinterpret_cast<KeyType*>(data_ + block_size_ * block_id);
    ValueType* blk_v = reinterpret_cast<ValueType*>(data_ + block_size_ * block_id + rows * sizeof(KeyType));
    auto inner_id = faiss::binary_search_eq(blk_k, rows, key);

    if (inner_id == -1) {
        return {-1};
    } else if (more_res == false) {
        return {blk_v[inner_id]};
    } else {
        std::vector<ValueType> res;
        for (; inner_id < rows; inner_id++) {
            if (key == blk_k[inner_id]) {
                res.emplace_back(blk_v[inner_id]);
            } else {
                break;
            }
        }
        return res;
    }
}

template <typename IN_HASH_TYPE>
Status
MinHashIndex<IN_HASH_TYPE>::BuildAndSave(MinHashIndexBuildParams* params) {
    if (params == nullptr) {
        LOG_KNOWHERE_ERROR_ << "build parameters is null.";
        return Status::invalid_args;
    }
    std::unique_ptr<IN_HASH_TYPE[]> data = nullptr;
    size_t ntotal, dim;
    load_binary_file<IN_HASH_TYPE>(params->data_path, data, ntotal, dim);
    if (dim % params->band != 0) {
        LOG_KNOWHERE_ERROR_ << "dim % params.band != 0";
        return Status::invalid_args;
    }
    size_t band_index_n = params->band;
    size_t block_size = params->block_size;
    std::shared_ptr<KVPair[]> total_kv_pair = gen_transposed_hash_kv(data.get(), ntotal, dim, band_index_n);
    sort_kv(total_kv_pair, ntotal, band_index_n);

    faiss::BlockFileIOWriter writer(params->index_file_path.c_str(), params->block_size);
    size_t data_pos = -1;
    if (params->has_raw_data) {
        data_pos = writer.tellg();
        writer.flush_and_write((char*)data.get(), ntotal * dim * sizeof(IN_HASH_TYPE));
    }
    std::vector<size_t> band_index_ofs(band_index_n);
    for (size_t index_i = 0; index_i < band_index_n; index_i++) {
        band_index_ofs[index_i] =
            MinHashBandIndex::FormatAndSave(writer, total_kv_pair.get() + index_i * ntotal, block_size, ntotal);
    }

    // write file header
    {
        MemoryIOWriter header_writer;
        writeBinaryPOD(header_writer, ntotal);
        writeBinaryPOD(header_writer, dim);
        writeBinaryPOD(header_writer, block_size);
        writeBinaryPOD(header_writer, band_index_n);
        writeBinaryPOD(header_writer, data_pos);
        header_writer.write((char*)band_index_ofs.data(), band_index_ofs.size() * sizeof(size_t));
        writer.write_header((char*)header_writer.data_, header_writer.rp_);
        if (header_writer.data_) {
            delete[] header_writer.data_;
        }
    }
    return Status::success;
}

template <typename IN_HASH_TYPE>
Status
MinHashIndex<IN_HASH_TYPE>::Load(MinHashIndexLoadParams* params) {
    if (params == nullptr) {
        LOG_KNOWHERE_ERROR_ << "load parameters is null.";
        return Status::invalid_args;
    }
    auto reader = FileReader(params->index_file_path);
    readBinaryPOD(reader, this->ntotal_);
    readBinaryPOD(reader, this->dim_);
    readBinaryPOD(reader, block_size_);
    readBinaryPOD(reader, band_);
    size_t data_pos;
    readBinaryPOD(reader, data_pos);
    if (data_pos == -1) {
        this->with_raw_data_ = false;
    } else {
        this->with_raw_data_ = true;
    }

    if (!params->hash_code_in_memory || this->with_raw_data_) {
        auto f = std::unique_ptr<FILE, decltype(&fclose)>(fopen(params->index_file_path.c_str(), "r"), &fclose);
        struct stat s;
        fstat(fileno(f.get()), &s);
        this->file_size_ = s.st_size;
        this->mmap_data_ = static_cast<char*>(mmap(NULL, file_size_, PROT_READ, MAP_SHARED, fileno(f.get()), 0));
        if (mmap_data_ == MAP_FAILED) {
            LOG_KNOWHERE_ERROR_ << "fail to mmap data ." << errno << " " << strerror(errno) << std::endl;
            return Status::disk_file_error;
        }
    } else {
        this->mmap_data_ = nullptr;
    }
    if (this->with_raw_data_) {
        raw_data_ = (IN_HASH_TYPE*)(mmap_data_ + data_pos);
    }
    band_index_ = std::make_unique<MinHashBandIndex[]>(band_);
    std::vector<size_t> band_index_ofs(band_);
    reader.read((char*)band_index_ofs.data(), band_index_ofs.size() * sizeof(size_t));
    if (params->global_bloom_filter) {
        bloom_ = std::vector<BloomFilter<KeyType>>(1, BloomFilter<KeyType>(this->ntotal_, params->false_positive_prob));
    } else {
        bloom_ =
            std::vector<BloomFilter<KeyType>>(band_, BloomFilter<KeyType>(this->ntotal_, params->false_positive_prob));
    }
    auto band_mmap_addr = params->hash_code_in_memory ? nullptr : this->mmap_data_;
    for (size_t i = 0; i < band_; i++) {
        reader.seek(band_index_ofs[i]);
        band_index_[i].Load(reader, this->ntotal_, band_mmap_addr, bloom_[i % bloom_.size()], i == 0);
    }
    is_loaded_ = true;
    return Status::success;
}

template <typename IN_HASH_TYPE>
void
MinHashIndex<IN_HASH_TYPE>::Search(const char* query, float* distances, Idx* labels) {
    *distances = 0;
    *labels = -1;
    for (auto i = 0; i < band_; i++) {
        const auto hash = get_hash_key((const IN_HASH_TYPE*)query, this->dim_, band_, i);
        auto& band = band_index_[i];
        auto& bloom = bloom_[i % bloom_.size()];
        if (bloom.contains(hash)) {
            auto ids = band.Search(hash, with_raw_data_);
            for (auto& id : ids) {
                if (id != -1) {
                    auto dis = with_raw_data_ ? minhash_jaccard((const IN_HASH_TYPE*)query, raw_data_ + id * this->dim_,
                                                                this->dim_, band_)
                                              : 1;
                    if (dis != 0) {
                        *distances = 1;
                        *labels = id;
                        return;
                    }
                }
            }
        }
    }
    return;
}

template <typename IN_HASH_TYPE>
void
MinHashIndex<IN_HASH_TYPE>::BatchSearch(const char* query, size_t nq, float* distances, Idx* labels,
                                        std::shared_ptr<ThreadPool> pool) {
    std::vector<std::vector<KVPair>> q_band_hash(band_, std::vector<KVPair>());
    for (auto i = 0; i < nq; i++) {
        distances[i] = 0;
        labels[i] = -1;
        for (auto j = 0; j < band_; j++) {
            auto hash = get_hash_key((const IN_HASH_TYPE*)query + i * this->dim_, this->dim_, band_, j);
            if (bloom_[j % bloom_.size()].contains(hash)) {
                q_band_hash[j].emplace_back(KVPair{hash, i});
            }
        }
    }
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(band_);
    for (size_t b_i = 0; b_i < band_; b_i++) {
        //   std::cout<<"band i"<<b_i<<std::endl;
        futures.emplace_back(pool->push([&, &band = band_index_[b_i], &q_kv_list = q_band_hash[b_i]]() {
            for (auto& q_kv : q_kv_list) {
                //  std::cout <<"q_kv:"<<q_kv.Value<<std::endl;
                if (labels[q_kv.Value] != -1)
                    continue;
                auto hash = q_kv.Key;
                auto ids = band.Search(hash, with_raw_data_);
                //  float dis;
                //  Idx id;
                // MinHashIndex::Search(query + this->dim_ * q_kv.Value, dis, id);
                for (auto& id : ids) {
                    if (id != -1) {
                        auto dis = with_raw_data_
                                       ? minhash_jaccard((const IN_HASH_TYPE*)query + this->dim_ * q_kv.Value,
                                                         raw_data_ + id * this->dim_, this->dim_, band_)
                                       : 1;
                        if (dis != 0) {
                            distances[q_kv.Value] = 1;
                            labels[q_kv.Value] = id;
                        }
                    }
                }
            }
            return;
        }));
    }
    WaitAllSuccess(futures);
    return;
}
}  // namespace knowhere
#endif
