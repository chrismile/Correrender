/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2020, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CORRERENDER_SEMAPHORE_HPP
#define CORRERENDER_SEMAPHORE_HPP

#include <mutex>
#include <memory>
#include <condition_variable>
#include <atomic>

class Semaphore {
private:
    std::mutex _mutex;
    std::condition_variable _cv;
    unsigned long _count = 0;
    unsigned long _maxCount = 0;

public:
    Semaphore(unsigned long count = 0) : _count(count) {}

    void release(unsigned long n = 1) {
        {
            std::lock_guard<decltype(_mutex)> lock(_mutex);
            _count += n;
            _maxCount = std::max(_maxCount, _count);
        }
        _cv.notify_all();
    }

    void acquire(unsigned long n = 1) {
        while (true) {
            std::unique_lock<decltype(_mutex)> lock(_mutex);
            if (_count >= n) {
                _count -= n;
                break;
            }
            _cv.wait(lock);
        }
    }

    unsigned long peekCount() {
        std::lock_guard<decltype(_mutex)> lock(_mutex);
        return _count;
    }

    unsigned long getMaxCount() {
        std::lock_guard<decltype(_mutex)> lock(_mutex);
        return _maxCount;
    }
};
using unique_semaphore = std::unique_ptr<Semaphore>;

#endif
