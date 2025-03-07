/*!
 * \file
 *
 * Copyright (c) 2024 Mark Mizzi
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <cctype>
#include <iostream>
#include <iterator>
#include <list>

#include "assertalways.h"
#include "vector.h"

namespace libbase
{

enum range_step_method {
    ARITHMETIC,
    GEOMETRIC,
};

/*!
 * \brief   Range class.
 * \author  Mark Mizzi
 *
 * Represents a range of doubles that are specified using start, stop, step
 * There are multiple stepping methods, including arithmetic, geometric, and so
 * on.
 */
class range
{
    // Allow iterator to access private fields such as start, stop, step.
    friend class iterator;

private:
    double start, stop, step;
    range_step_method step_method;

public:
    range(double start,
          double stop,
          double step,
          range_step_method step_method = range_step_method::ARITHMETIC)
        : start(start), stop(stop), step(step), step_method(step_method)
    {
    }

    range()
        : start(0.0), stop(0.0), step(0.0),
          step_method(range_step_method::ARITHMETIC)
    {
    }

    /*!
     * \brief   Range iterator.
     * \author  Mark Mizzi
     *
     * This is an iterator object that allows us to iterate over the range of
     * values specified by the \ref range object. Typically created using \ref
     * begin() or \ref end()
     */
    class iterator
    {
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = int;
        using value_type = double;
        using pointer = const double*;
        using reference = const double&;

    private:
        range* it_range;
        /// @brief Current value held by iterator.
        double curr_value;

    public:
        iterator(range& it_range)
            : it_range(&it_range), curr_value(it_range.start)
        {
        }
        iterator(range& it_range, double value)
            : it_range(&it_range), curr_value(value)
        {
        }

        // ensure that default copy and assign by copy operators are created.
        iterator(const iterator&) = default;
        iterator& operator=(const iterator&) = default;

        reference operator*() const { return curr_value; }
        pointer operator->() const { return &curr_value; }

        // Prefix increment
        iterator& operator++()
        {
            switch (it_range->step_method) {
            case range_step_method::ARITHMETIC:
                curr_value += it_range->step;
                break;
            case range_step_method::GEOMETRIC:
                curr_value *= it_range->step;
                break;
            }
            return *this;
        }

        // Postfix increment
        iterator operator++(int)
        {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        // Prefix decrement
        iterator& operator--()
        {
            switch (it_range->step_method) {
            case range_step_method::ARITHMETIC:
                curr_value -= it_range->step;
                break;
            case range_step_method::GEOMETRIC:
                curr_value /= it_range->step;
                break;
            }
            return *this;
        }

        // Postfix decrement
        iterator operator--(int)
        {
            iterator tmp = *this;
            --(*this);
            return tmp;
        }

        //! \section Comparision operators
        friend bool operator<(const iterator& a, const iterator& b)
        {
            // cannot compare iterators from different range objects
            assertalways(&a.it_range == &b.it_range);
            return a.curr_value < b.curr_value;
        };
        friend bool operator<=(const iterator& a, const iterator& b)
        {
            // cannot compare iterators from different range objects
            assertalways(&a.it_range == &b.it_range);
            return a.curr_value <= b.curr_value;
        };
        friend bool operator>=(const iterator& a, const iterator& b)
        {
            // cannot compare iterators from different range objects
            assertalways(&a.it_range == &b.it_range);
            return a.curr_value >= b.curr_value;
        };
        friend bool operator>(const iterator& a, const iterator& b)
        {
            // cannot compare iterators from different range objects
            assertalways(&a.it_range == &b.it_range);
            return a.curr_value > b.curr_value;
        };
    };

    /*! \brief Returns iterator whose current value is the beginning of the
     * range.
     */
    iterator begin() { return iterator(*this); }
    /*! \brief Returns iterator whose current value is the end of the
     * range.
     */
    iterator end() { return iterator(*this, this->stop); }

    /*! \brief Parse range from input stream.
     */
    friend std::istream& operator>>(std::istream& is, range& r)
    {
        /// read r from stream
        // parse start, step, stop
        double range_spec[3];
        for (int i = 0; i < 3; i++) {
            is >> range_spec[i];
            char c = is.get();
            if (c != ':') {
                is.setstate(std::ios::failbit);
                failwith(std::string(
                             "Invalid range specified, expected : and got ") +
                         std::to_string(c));
            }
        }
        // parse step method
        std::string step_method_str;
        while (!std::isspace(is.peek()))
            step_method_str.push_back(is.get());
        range_step_method step_method =
            range_step_method::ARITHMETIC; // set a default to please compiler.
        if (step_method_str == "arithmetic") {
            step_method = range_step_method::ARITHMETIC;
        } else if (step_method_str == "geometric") {
            step_method = range_step_method::GEOMETRIC;
        } else {
            is.setstate(std::ios::failbit);
            failwith(std::string("Did not get valid step method, got ") +
                     step_method_str);
        }

        r.start = range_spec[0];
        r.step = range_spec[1];
        r.stop = range_spec[2];
        r.step_method = step_method;

        return is;
    }
};

/*!
 * \brief   Multi-range iterator.
 * \author  Mark Mizzi
 *
 * Represents an iterator object that groups together a lot of \ref
 * range::iterator objects, and iterates over combinations of the values
 * returned by the iterator objects in lexicographical order.
 */
class multi_range_iterator
{
private:
    /*! \brief Vector of ranges we are iterating over, we need to keep these to
     * create iterators from.
     */
    libbase::vector<range>& ranges;
    /*! \brief List of iterators, this tracks our current position in the
     * overall iteration
     */
    std::list<range::iterator> iterators;

public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = int;
    using value_type = libbase::vector<double>;

private:
    /*! \brief Private constructor which does not initialize iterator list
     */
    multi_range_iterator(libbase::vector<range>& ranges) : ranges(ranges) {}

public:
    /*! \brief Public factory method that constructs a \ref multi_range_iterator
     * object and initializes each iterator to point to beginning of its
     * respective \ref range
     */
    static multi_range_iterator begin(libbase::vector<range>& ranges)
    {
        multi_range_iterator it(ranges);
        for (int i = 0; i < ranges.size(); i++) {
            it.iterators.push_back(ranges(i).begin());
        }
        return it;
    }

    /*! \brief Public factory method that constructs a \ref multi_range_iterator
     * object and initializes each iterator to point to end of its
     * respective \ref range
     */
    static multi_range_iterator end(libbase::vector<range>& ranges)
    {
        multi_range_iterator it(ranges);
        for (int i = 0; i < ranges.size(); i++) {
            it.iterators.push_back(ranges(i).end());
        }
        return it;
    }

    value_type operator*() const
    {
        value_type vals;
        vals.init(ranges.size());
        int i = 0;
        for (auto it = iterators.begin(); i < ranges.size(); i++, ++it)
            vals(i) = **it;
        return vals;
    }

    // Pre-fix increment
    multi_range_iterator& operator++()
    {
        int i = ranges.size() - 1;
        for (auto it = iterators.rbegin(); it != iterators.rend(); ++it, --i) {
            ++*it;
            if (*it >= ranges(i).end()) {
                *it = ranges(i).begin();
            } else {
                break;
            }
        }
        return *this;
    }

    // Postfix increment
    multi_range_iterator operator++(int)
    {
        multi_range_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    //! \section Comparision operators, these are defined according to
    //! lexicographical order.
    friend bool operator<(const multi_range_iterator& a,
                          const multi_range_iterator& b)
    {
        assertalways(a.iterators.size() == b.iterators.size());

        for (auto it1 = a.iterators.begin(), it2 = b.iterators.begin();
             it1 != a.iterators.end();
             ++it1, ++it2) {
            if (*it1 < *it2) {
                return true;
            } else if (*it1 > *it2) {
                return false;
            }
            // in else case *it1 == *it2, so we move on to compare next it.
        }
        return false; // case where a == b
    };
    friend bool operator<=(const multi_range_iterator& a,
                           const multi_range_iterator& b)
    {
        assertalways(a.iterators.size() == b.iterators.size());

        for (auto it1 = a.iterators.begin(), it2 = b.iterators.begin();
             it1 != a.iterators.end();
             ++it1, ++it2) {
            if (*it1 < *it2) {
                return true;
            } else if (*it1 > *it2) {
                return false;
            }
            // in else case *it1 == *it2, so we move on to compare next it.
        }
        return true; // case where a == b
    };
    friend bool operator>=(const multi_range_iterator& a,
                           const multi_range_iterator& b)
    {
        assertalways(a.iterators.size() == b.iterators.size());

        for (auto it1 = a.iterators.begin(), it2 = b.iterators.begin();
             it1 != a.iterators.end();
             ++it1, ++it2) {
            if (*it1 > *it2) {
                return true;
            } else if (*it1 < *it2) {
                return false;
            }
            // in else case *it1 == *it2, so we move on to compare next it.
        }
        return true; // case where a == b
    };
    friend bool operator>(const multi_range_iterator& a,
                          const multi_range_iterator& b)
    {
        assertalways(a.iterators.size() == b.iterators.size());

        for (auto it1 = a.iterators.begin(), it2 = b.iterators.begin();
             it1 != a.iterators.end();
             ++it1, ++it2) {
            if (*it1 > *it2) {
                return true;
            } else if (*it1 < *it2) {
                return false;
            }
            // in else case *it1 == *it2, so we move on to compare next it.
        }
        return false; // case where a == b
    };
};

} // end namespace libbase