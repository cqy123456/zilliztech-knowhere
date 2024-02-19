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

#include <string>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/diskann/diskann_config.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/expected.h"
#include "knowhere/factory.h"
#include "knowhere/utils.h"
#include "utils.h"
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif
#include <fstream>

namespace {
std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kIPIndexDir = kDir + "/ip_index";
std::string kCOSINEIndexDir = kDir + "/cosine_index";
std::string kL2IndexPrefix = kL2IndexDir + "/l2";
std::string kIPIndexPrefix = kIPIndexDir + "/ip";
std::string kCOSINEIndexPrefix = kCOSINEIndexDir + "/cosine";

constexpr uint32_t kNumRows = 1000;
constexpr uint32_t kNumQueries = 10;
constexpr uint32_t kDim = 128;
constexpr uint32_t kLargeDim = 1536;
constexpr uint32_t kK = 10;
constexpr float kKnnRecall = 0.9;
constexpr float kL2RangeAp = 0.9;
constexpr float kIpRangeAp = 0.9;
constexpr float kCosineRangeAp = 0.9;

template <typename DataType>
void
WriteRawDataToDisk(const std::string data_path, const DataType* raw_data, const uint32_t num, const uint32_t dim) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&num, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)raw_data, sizeof(DataType) * num * dim);
    writer.close();
}

}  // namespace

template <typename DataType>
inline void
BaseSearchTest() {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = 36;
        json["beamwidth"] = 8;
        return json;
    };

    auto range_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["beamwidth"] = 8;
        json["min_k"] = 10;
        json["max_k"] = 8000;
        return json;
    };

    auto fp32_query_ds = GenDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    auto fp32_base_ds = GenDataSet(kNumRows, kDim, 30);
    knowhere::DataSetPtr base_ds(fp32_base_ds);
    knowhere::DataSetPtr query_ds(fp32_query_ds);
    if (!std::is_same_v<knowhere::fp32, DataType>) {
        base_ds = knowhere::data_type_conversion<knowhere::fp32, DataType>(*fp32_base_ds);
        query_ds = knowhere::data_type_conversion<knowhere::fp32, DataType>(*fp32_query_ds);
    }
    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();

        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<DataType>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }

    SECTION("Test search and range search") {
        std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;
        // build process
        {
            knowhere::DataSet* ds_ptr = nullptr;
            auto diskann = knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack);
            auto build_json = build_gen().dump();
            knowhere::Json json = knowhere::Json::parse(build_json);
            diskann.Build(*ds_ptr, json);
            diskann.Serialize(binset);
        }
        {
            // knn search
            auto diskann = knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack);
            diskann.Deserialize(binset, deserialize_json);

            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(*query_ds, knn_json, nullptr);
            REQUIRE(res.error() == knowhere::Status::success);
            REQUIRE(res.has_value());
            auto knn_recall = GetKNNRecall(*knn_gt_ptr, *res.value());
            REQUIRE(knn_recall > kKnnRecall);

            // knn search without cache file
            {
                std::string cached_nodes_file_path =
                    std::string(build_gen()["index_prefix"]) + std::string("_cached_nodes.bin");
                if (fs::exists(cached_nodes_file_path)) {
                    fs::remove(cached_nodes_file_path);
                }
                auto diskann_tmp =
                    knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack);
                diskann_tmp.Deserialize(binset, deserialize_json);
                auto knn_search_json = knn_search_gen().dump();
                knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
                auto res = diskann_tmp.Search(*query_ds, knn_json, nullptr);
                REQUIRE(res.has_value());
                REQUIRE(GetKNNRecall(*knn_gt_ptr, *res.value()) >= kKnnRecall);
            }

            // knn search with bitset
            std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
                GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
            const auto bitset_percentages = {0.4f, 0.98f};
            const auto bitset_thresholds = {-1.0f, 0.9f};
            for (const float threshold : bitset_thresholds) {
                knn_json["filter_threshold"] = threshold;
                for (const float percentage : bitset_percentages) {
                    for (const auto& gen_func : gen_bitset_funcs) {
                        auto bitset_data = gen_func(kNumRows, percentage * kNumRows);
                        knowhere::BitsetView bitset(bitset_data.data(), kNumRows);
                        auto results = diskann.Search(*query_ds, knn_json, bitset);
                        auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, knn_json, bitset);
                        float recall = GetKNNRecall(*gt.value(), *results.value());
                        if (percentage == 0.98f) {
                            REQUIRE(recall >= 0.9f);
                        } else {
                            REQUIRE(recall >= kKnnRecall);
                        }
                    }
                }
            }

            // range search process
            auto range_search_json = range_search_gen().dump();
            knowhere::Json range_json = knowhere::Json::parse(range_search_json);
            auto range_search_res = diskann.RangeSearch(*query_ds, range_json, nullptr);
            REQUIRE(range_search_res.has_value());
            auto ap = GetRangeSearchRecall(*range_search_gt_ptr, *range_search_res.value());
            float standard_ap = metric_range_ap_map[metric_str];
            REQUIRE(ap > standard_ap);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

// This test case only check L2
template <typename DataType>
inline void
GetVectorByIdsTest() {
    auto version = GenTestVersionList();
    for (const uint32_t dim : {kDim, kLargeDim}) {
        fs::remove_all(kDir);
        fs::remove(kDir);
        REQUIRE_NOTHROW(fs::create_directories(kL2IndexDir));

        auto base_gen = [&] {
            knowhere::Json json;
            json[knowhere::meta::RETRIEVE_FRIENDLY] = true;
            json["dim"] = dim;
            json["metric_type"] = knowhere::metric::L2;
            json["k"] = kK;
            return json;
        };

        auto build_gen = [&]() {
            knowhere::Json json = base_gen();
            json["index_prefix"] = kL2IndexPrefix;
            json["data_path"] = kRawDataPath;
            json["max_degree"] = 5;
            json["search_list_size"] = kK;
            json["pq_code_budget_gb"] = sizeof(DataType) * dim * kNumRows * 0.125 / (1024 * 1024 * 1024);
            json["build_dram_budget_gb"] = 32.0;
            return json;
        };

        auto fp32_query_ds = GenDataSet(kNumQueries, dim, 42);
        auto fp32_base_ds = GenDataSet(kNumRows, dim, 30);
        knowhere::DataSetPtr base_ds(fp32_base_ds);
        knowhere::DataSetPtr query_ds(fp32_query_ds);
        if (!std::is_same_v<knowhere::fp32, DataType>) {
            base_ds = knowhere::data_type_conversion<knowhere::fp32, DataType>(*fp32_base_ds);
            query_ds = knowhere::data_type_conversion<knowhere::fp32, DataType>(*fp32_query_ds);
        }
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, dim);

        std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);

        knowhere::DataSet* ds_ptr = nullptr;
        auto diskann = knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack);
        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        auto build_stat = diskann.Build(*ds_ptr, json);
        REQUIRE(build_stat == knowhere::Status::success);
        knowhere::BinarySet binset;
        diskann.Serialize(binset);
        {
            std::vector<double> cache_sizes = {0,
                                               1.0f * sizeof(DataType) * dim * kNumRows * 0.125 / (1024 * 1024 * 1024)};
            for (const auto cache_size : cache_sizes) {
                auto deserialize_gen = [&base_gen, cache = cache_size]() {
                    knowhere::Json json = base_gen();
                    json["index_prefix"] = kL2IndexPrefix;
                    json["search_cache_budget_gb"] = cache;
                    return json;
                };
                knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                auto index =
                    knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack);
                auto ret = index.Deserialize(binset, deserialize_json);
                REQUIRE(ret == knowhere::Status::success);
                std::vector<double> ids_sizes = {1, kNumRows * 0.2, kNumRows * 0.7, kNumRows};
                for (const auto ids_size : ids_sizes) {
                    std::cout << "Testing dim = " << dim << ", cache_size = " << cache_size
                              << ", ids_size = " << ids_size << std::endl;
                    auto ids_ds = GenIdsDataSet(ids_size, ids_size);
                    auto results = index.GetVectorByIds(*ids_ds);
                    REQUIRE(results.has_value());
                    auto xb = (DataType*)base_ds->GetTensor();
                    auto data = (DataType*)results.value()->GetTensor();
                    for (size_t i = 0; i < ids_size; ++i) {
                        auto id = ids_ds->GetIds()[i];
                        for (size_t j = 0; j < dim; ++j) {
                            REQUIRE(data[i * dim + j] == xb[id * dim + j]);
                        }
                    }
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test DiskANN Base ", "[diskann]") {
    BaseSearchTest<knowhere::fp32>();
    BaseSearchTest<knowhere::bf16>();
    BaseSearchTest<knowhere::fp16>();
}

TEST_CASE("Test DiskANN GetVectorByIds", "[diskann]") {
    GetVectorByIdsTest<knowhere::fp32>();
    GetVectorByIdsTest<knowhere::fp16>();
    GetVectorByIdsTest<knowhere::bf16>();
}
