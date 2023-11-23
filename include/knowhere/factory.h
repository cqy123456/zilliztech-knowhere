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

#ifndef INDEX_FACTORY_H
#define INDEX_FACTORY_H

#include <functional>
#include <string>
#include <unordered_map>

#include "knowhere/index.h"

namespace knowhere {
class IndexFactory {
 public:
    template <typename DataType>
    Index<IndexNode>
    Create(const std::string& name, const int32_t& version, const Object& object = nullptr);
    template <typename DataType>
    const IndexFactory&
    Register(const std::string& name, std::function<Index<IndexNode>(const int32_t& version, const Object&)> func);
    static IndexFactory&
    Instance();

 private:
    struct FunMapValueBase {};
    template <typename T1>
    struct FunMapValue : FunMapValueBase {
     public:
        FunMapValue(std::function<T1(const int32_t&, const Object&)>& input) : fun_value(input) {
        }
        std::function<T1(const int32_t&, const Object&)> fun_value;
    };
    typedef std::map<std::string, FunMapValueBase*> FuncMap;
    IndexFactory();
    static FuncMap&
    MapInstance();
    template <typename DataType>
    std::string
    GetMapKey(const std::string& name);
};

#define KNOWHERE_CONCAT(x, y) index_factory_ref_##x##y
#define KNOWHERE_CONCAT_STR(x, y) #x "_" #y
#define KNOWHERE_REGISTER_GLOBAL(name, func, data_type) \
    const IndexFactory& KNOWHERE_CONCAT(name, data_type) = IndexFactory::Instance().Register<data_type>(#name, func)
#define KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, data_type, ...)                  \
    KNOWHERE_REGISTER_GLOBAL(                                                              \
        name,                                                                              \
        [](const int32_t& version, const Object& object) {                                 \
            return (Index<index_node<data_type, ##__VA_ARGS__>>::Create(version, object)); \
        },                                                                                 \
        data_type)
#define KNOWHERE_MOCK_REGISTER_GLOBAL(name, index_node, data_type, mock_data_type, ...)         \
    KNOWHERE_REGISTER_GLOBAL(                                                                   \
        name,                                                                                   \
        [](const int32_t& version, const Object& object) {                                      \
            return (Index<IndexNodeDataMockWrapper<data_type>>::Create(                         \
                std::make_unique<index_node<mock_data_type, ##__VA_ARGS__>>(version, object))); \
        },                                                                                      \
        data_type)
}  // namespace knowhere

#endif /* INDEX_FACTORY_H */
