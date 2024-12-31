//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef DATA_VIEW_INDEX_NODE_CONFIG_H
#define DATA_VIEW_INDEX_NODE_CONFIG_H
#include "knowhere/config.h"
#include "index/ivf/ivf_config.h"
namespace knowhere {
class IndexRefineConfig : public Config {
    CFG_FLOAT refine_ratio;
    KNOHWERE_DECLARE_CONFIG(IndexRefineConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_ratio)
            .set_default(1.0f)
            .description("search refine_ratio * k results then refine")
            .for_search();
    }
};
class BaseOffsetIndexRefineConfig: public BaseConfig,IndexRefineConfig {};
class OffsetIndexSCANNConfig: public  ScannConfig, IndexRefineConfig {};
} // namespace knowhere
#endif
