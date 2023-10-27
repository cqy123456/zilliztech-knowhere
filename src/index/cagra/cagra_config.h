/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "knowhere/config.h"
#include  <set>
namespace knowhere {
constexpr const CFG_INT::value_type kInterDegreeDefaultValue = 128;
constexpr const CFG_INT::value_type kGraphDegreeDefaultValue = 64;
class CagraConfig : public BaseConfig {
 public:
    CFG_INT intermediate_graph_degree;
    CFG_INT graph_degree;
    CFG_INT itopk_size;
    CFG_INT max_queries;
    CFG_STRING algo;
    CFG_INT team_size;
    CFG_INT search_width;
    CFG_INT min_iterations;
    CFG_INT max_iterations;
    CFG_INT thread_block_size;
    CFG_STRING hashmap_mode;
    CFG_INT hashmap_min_bitlen;
    CFG_FLOAT hashmap_max_fill_rate;

    KNOHWERE_DECLARE_CONFIG(CagraConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(k)
            .set_default(10)
            .description("search for top k similar vector.")
            .set_range(1, 1024)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(intermediate_graph_degree)
            .description("degree of input graph for pruning.")
            .allow_empty_without_default()
            .for_train()
            .set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(graph_degree)
            .description("degree of output graph.")
            .allow_empty_without_default()
            .for_train()
            .set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(itopk_size)
            .set_default(64)
            .description("number of intermediate search results retained during the search.")
            .for_search()
            .set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(max_queries).description("query batch size.").set_default(1).for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(algo)
            .set_default("AUTO")
            .description("Which search implementation to use.")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_width)
            .description("Number of graph nodes to select as the starting point in each search iteration.")
            .set_default(1)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(min_iterations)
            .description("Lower limit of search iterations.")
            .set_default(0)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_iterations)
            .description("Upper limit of search iterations. Auto select when 0.")
            .set_default(0)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(team_size)
            .description("Number of threads used to calculate a single distance. 4, 8, 16, or 32.")
            .set_default(0)
            .set_range(0, 32)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(thread_block_size)
            .description("Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.")
            .set_default(0)
            .set_range(0, 1024)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(hashmap_mode)
            .description("Hashmap type. Auto selection when AUTO.")
            .set_default("AUTO")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(hashmap_min_bitlen)
            .description("Lower limit of hashmap bit length. More than 8..")
            .set_default(0)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(hashmap_max_fill_rate)
            .description("Upper limit of hashmap fill rate. More than 0.1, less than 0.9.")
            .set_default(0.5)
            .set_range(0.1, 0.9)
            .for_search();
    }
    inline Status
    CheckAndAdjustForBuild() override {
        if (metric_type.value() != "L2") {
            LOG_KNOWHERE_ERROR_ << "cagra only support L2";
            return Status::invalid_metric_type;
        }
        if (!intermediate_graph_degree.has_value() && !graph_degree.has_value()) {
            intermediate_graph_degree = kInterDegreeDefaultValue;
            graph_degree = kGraphDegreeDefaultValue;
        } else if (!graph_degree.has_value()) {
            graph_degree = uint32_t(std::ceil(intermediate_graph_degree.value()/2.0));
        } else if (!intermediate_graph_degree.has_value()) {
            intermediate_graph_degree = graph_degree.value() * 2;
        } else if (graph_degree.value() > intermediate_graph_degree.value()) {
            LOG_KNOWHERE_ERROR_ << "intermediate_graph_degree should be larger than graph_degree";
            return Status::out_of_range_in_json;
        }
        return Status::success;
    }

    inline Status
    CheckAndAdjustForSearch(std::string* err_msg) override {
        std::set<int32_t> legal_team_size = {0, 4, 8, 16, 32};
        std::set<int32_t> legal_thread_block_size = {0, 64, 128, 256, 512, 1024};
        if (metric_type.has_value() && metric_type.value() != "L2") {
            *err_msg = "cagra only support L2";
            LOG_KNOWHERE_ERROR_ << *err_msg;
            return Status::invalid_metric_type;
        }
        if (team_size.has_value() && legal_team_size.find(team_size.value()) == legal_team_size.end()) {
            *err_msg = 
                "team_size(" + std::to_string(team_size.value()) + ") should be in [0, 4, 8, 16, 32]";
            LOG_KNOWHERE_ERROR_ << *err_msg;
            return Status::out_of_range_in_json;
        }
        if (thread_block_size.has_value() && legal_thread_block_size.find(thread_block_size.value()) == legal_team_size.end()) {
            *err_msg = 
                "thread_block_size(" + std::to_string(thread_block_size.value()) + ") should be in [0, 64, 128, 256, 512, 1024]";
            LOG_KNOWHERE_ERROR_ << *err_msg;
            return Status::out_of_range_in_json;
        }
        return Status::success;
    }
};

}  // namespace knowhere
