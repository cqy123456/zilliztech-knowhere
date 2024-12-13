  
// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
namespace knowhere {
class ThreadSafeStateController {
   public:
    bool start() {
        std::unique_lock<std::mutex> guard(status_mtx);
        if (status.load() ==
            ThreadSafeStateController::Status::KILLED) {
            status.store(
              ThreadSafeStateController::Status::DONE);
          return false;
        }
        status.store(
            ThreadSafeStateController::Status::DOING);
          return true;
    }

    void complete() {
        std::unique_lock<std::mutex> guard(status_mtx);
        status.store(ThreadSafeStateController::Status::DONE);
        cond.notify_one();
    }

    bool is_cancel() {
      std::unique_lock<std::mutex> guard(status_mtx);
      return (ThreadSafeStateController::Status::STOPPING == status.load());
    }

    void cancel() {
      std::unique_lock<std::mutex> guard(status_mtx);
      if (status.load() == ThreadSafeStateController::Status::DONE) {
        return;
      }
      if (status.load() == ThreadSafeStateController::Status::NONE) {
        status.store(ThreadSafeStateController::Status::KILLED);
        return;
      }
      status.store(ThreadSafeStateController::Status::STOPPING);
      if (status.load() != ThreadSafeStateController::Status::DONE) {
        cond.wait(guard);
      }
    }

    private:
      enum class Status {
        NONE,
        DOING,
        STOPPING,
        DONE,
        KILLED,
      };
      std::atomic<Status>     status = Status::NONE;
      std::condition_variable cond;
      std::mutex              status_mtx;
  };
using ThreadSafeStateControllerPtr = std::shared_ptr<ThreadSafeStateController>; 
} // namespace knowhere