/* Copyright 2020 The Blueoil Authors. All Rights Reserved.  
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/
#pragma once

// Notice(kajihara):
//  This is now a helper header for converter-unit-test.

#include <cinttypes>
#include <iterator>
#include <algorithm>

namespace helper{
    namespace impl{
        template<class T>
        class irange_t{
            T first, last;
            
        public:
            constexpr irange_t(T _first, T _last):
            first(_first),
            last(_last){}

            class iterator{
                T value;
            public:
                using difference_type   = std::ptrdiff_t;
                using value_type        = T;
                using pointer           = T*;
                using reference         = T&;
                using iterator_category = std::input_iterator_tag;

                constexpr iterator(T _value):
                value(_value){}

                constexpr auto& operator++(){
                    ++value;
                    return *this;
                }
                constexpr auto operator++(int){
                    auto ret = *this;
                    ++value;
                    return ret;
                }

                constexpr bool operator!=(iterator ite)const{
                    return value != ite.value;
                }

                constexpr T operator*()const{
                    return value;
                }

                constexpr T operator[](T v)const{
                    return value + v;
                }
            };
            
            constexpr auto begin()const{
                return iterator(first);
            }

            constexpr auto end()const{
                return iterator(last);
            }
        };
    }

    template<class T>constexpr auto irange(T first, T last){
        return impl::irange_t<T>(first, std::max(first, last));
    }
    
    template<class T>constexpr auto irange(T last){
        return irange(T(), last);
    }
} // namespace helper