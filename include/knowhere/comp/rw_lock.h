//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.
#ifndef KNOWHERE_RW_LOCK_H
#define KNOWHERE_RW_LOCK_H
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
/*
FairRWLock is a fair MultiRead-SingleWrite lock
*/
namespace knowhere {
class FairRWLock {
public:
    FairRWLock() : id_counter_(0), readers(0), writer(false) {}

    void LockRead() {
        std::unique_lock<std::mutex> lk(mtx);
        auto id = id_counter_++;

        read_cv.wait(lk, [this, id]() {
            return !writer && (write_requests.empty() || write_requests.front() > id);
        });

        ++readers;
    }

    void UnLockRead() {
        std::unique_lock<std::mutex> lk(mtx);
        if (--readers == 0 && !write_requests.empty()) {
            write_cv.notify_one();
        }
    }

    void LockWrite() {
        std::unique_lock<std::mutex> lk(mtx);
        auto id =  id_counter_++;
        write_requests.push(id);

        write_cv.wait(lk, [this, id]() {
            return !writer && readers == 0 && write_requests.front() == id;
        });

        write_requests.pop();
        writer = true;
    }

    void UnLockWrite() {
        std::unique_lock<std::mutex> lk(mtx);
        writer = false;
        if (!write_requests.empty()) {
            write_cv.notify_one();
        } else {
            read_cv.notify_all();
        }
    }

private:
    uint64_t id_counter_ = 0;
    std::mutex mtx;
    std::condition_variable read_cv;
    std::condition_variable write_cv;
    int readers;
    bool writer;
    std::queue<uint64_t> write_requests;
};

class FairReadLockGuard {
 public:
    explicit FairReadLockGuard(FairRWLock& lock) : lock_(lock) {
        lock_.LockRead();
    }

    ~FairReadLockGuard() {
        lock_.UnLockRead();
    }

 private:
    FairRWLock& lock_;
};

class FairWriteLockGuard {
 public:
    explicit FairWriteLockGuard(FairRWLock& lock) : lock_(lock) {
        lock_.LockWrite();
    }

    ~FairWriteLockGuard() {
        lock_.UnLockWrite();
    }

 private:
    FairRWLock& lock_;
};
}  // namespace knowhere
#endif
