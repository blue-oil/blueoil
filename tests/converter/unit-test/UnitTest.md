# Converter Unit Test

This directory is for unit test of converter C++ runtime.
This is rerated to issue #743(https://github.com/blue-oil/blueoil/issues/743).
For detail, see the issue.

## Dummy Function Idea

For unit test, we must write dummy function definitions instead of true function definitions. But C++ has "One Definition Rule(ODR)"(for detail, see follow site https://en.cppreference.com/w/cpp/language/definition). My (kajihara's) idea for avoid ODR violation is follow.

- declare and define functions in namespace for unit test
- and write "using namespace ${the namespace name};" at end of the header

## Include Guards

"#pragma once" is useful but is not C++ standard. Using "ifndef", "define", and "endif" acts as include guard too, but this includes a risk of mistake due to a lot of description.

I made tool for replace "#pragma once" to "ifndef include guard" (https://gist.github.com/lm-kajihara/ae926ad3aa05f8608aaf69f0df2f43eb). I propose that use "#pragma once" at first, then use the tool.

## Helper Functions and Classes

"range/algorithm.hpp", "range/numeric.hpp", and "range/irange.hpp" are header files for range-based algorithm. These files should help writing unit tests.
