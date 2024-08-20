/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#ifndef CORRERENDER_LRUCACHE_HPP
#define CORRERENDER_LRUCACHE_HPP

#include <list>
#include <unordered_map>
#include <memory>
#include <type_traits>
#include "Volume/FieldAccess.hpp"

template<class T>
struct is_shared_ptr : std::false_type {};
template<class T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

/**
 * Last-recently-used (LRU) cache system.
 * For more details see: https://stackoverflow.com/questions/2504178/lru-cache-design
 */
template<class K, class V> class LRUCache {
private:
    using element = std::pair<K, V>;
    std::list<element> itemList;
    using list_element_iterator = typename std::list<element>::iterator;
    std::unordered_map<K, list_element_iterator> itemMap;

public:
    [[nodiscard]] bool empty() const {
        return itemMap.empty();
    }

    [[nodiscard]] bool exists(const K& key) const {
        return itemMap.find(key) != itemMap.end();
    }

    void push(const K& key, const V& value) {
        auto it = itemMap.find(key);
        if(it != itemMap.end()){
            itemList.erase(it->second);
            itemMap.erase(it);
        }
        itemList.push_front(std::make_pair(key, value));
        itemMap.insert(std::make_pair(key, itemList.begin()));
    }

    std::pair<K, V> pop(const K& key) {
        auto it = itemMap.find(key);
        itemList.splice(itemList.begin(), itemList, it->second);
        return *it->second;
    }

    std::pair<K, V> pop_last() {
        auto it = itemList.end();
        it--;
        std::pair<K, V> lastKeyValue = *it;
        itemMap.erase(it->first);
        itemList.pop_back();
        return lastKeyValue;
    }

    void remove_if(std::function<bool(std::pair<K, V>&)> predicate) {
        auto it = itemList.begin();
        while (it != itemList.end()) {
            if (predicate(*it)) {
                itemMap.erase(itemMap.find(it->first));
                it = itemList.erase(it);
            } else {
                it++;
            }
        }
    }

    void do_for_each(const std::function<void(const V&)>& functor) {
        for (auto& entry : itemList) {
            functor(entry.second);
        }
    }

};

#endif //CORRERENDER_LRUCACHE_HPP
