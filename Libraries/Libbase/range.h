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

enum RangeStepMethod {
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
    RangeStepMethod step_method;

public:
    range(double start,
          double stop,
          double step,
          RangeStepMethod step_method = RangeStepMethod::ARITHMETIC)
        : start(start), stop(stop), step(step), step_method(step_method)
    {
        // boundary checks to make sure that the range will (probably) not
        // iterate forever.
        if (step_method == RangeStepMethod::ARITHMETIC) {
            assertalways(step > 0 || stop <= start);
        } else if (step_method == RangeStepMethod::GEOMETRIC) {
            // we probably want to avoid non-monotonic sequences.
            assertalways(step >= 0);
            assertalways(step >= 1 || stop <= start);
        }
    }

    range()
        : start(0.0), stop(0.0), step(0.0),
          step_method(RangeStepMethod::ARITHMETIC)
    {
    }

    /*! \brief Return number of elements in the range object. */
    int count()
    {
        int cnt = 0;
        for (auto it = begin(); it <= end(); ++it)
            ++cnt;
        return cnt;
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
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = int;
        using value_type = double;
        using pointer = const double*;
        using reference = const double&;

    private:
        range* range_p;
        /// @brief Current value held by iterator.
        double curr_value;

    public:
        iterator(range& range_obj)
            : range_p(&range_obj), curr_value(range_obj.start)
        {
        }
        iterator(range& range_obj, double value)
            : range_p(&range_obj), curr_value(value)
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
            switch (range_p->step_method) {
            case RangeStepMethod::ARITHMETIC:
                curr_value += range_p->step;
                break;
            case RangeStepMethod::GEOMETRIC:
                curr_value *= range_p->step;
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
            switch (range_p->step_method) {
            case RangeStepMethod::ARITHMETIC:
                curr_value -= range_p->step;
                break;
            case RangeStepMethod::GEOMETRIC:
                curr_value /= range_p->step;
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
        bool operator<(const iterator& other) const
        {
            // cannot compare iterators from different range objects
            assertalways(this->range_p == other.range_p);
            if (this->range_p->start <= this->range_p->stop) {
                return this->curr_value < other.curr_value;
            } else {
                return this->curr_value > other.curr_value;
            }
        }
        bool operator<=(const iterator& other) const
        {
            // cannot compare iterators from different range objects
            assertalways(this->range_p == other.range_p);
            if (this->range_p->start <= this->range_p->stop) {
                return this->curr_value <= other.curr_value;
            } else {
                return this->curr_value >= other.curr_value;
            }
        }
        bool operator>=(const iterator& other) const
        {
            // cannot compare iterators from different range objects
            assertalways(this->range_p == other.range_p);
            if (this->range_p->start <= this->range_p->stop) {
                return this->curr_value >= other.curr_value;
            } else {
                return this->curr_value <= other.curr_value;
            }
        }
        bool operator>(const iterator& other) const
        {
            // cannot compare iterators from different range objects
            assertalways(this->range_p == other.range_p);
            if (this->range_p->start <= this->range_p->stop) {
                return this->curr_value > other.curr_value;
            } else {
                return this->curr_value < other.curr_value;
            }
        }
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
        std::string step_method_str;
        int c;
        while ((c = is.get()) != ':')
            step_method_str.push_back(c);

        if (step_method_str == "arithmetic") {
            r.step_method = RangeStepMethod::ARITHMETIC;
        } else if (step_method_str == "geometric") {
            r.step_method = RangeStepMethod::GEOMETRIC;
        } else {
            failwith(std::string("Did not get valid step method, got ") +
                     step_method_str);
        }

        is >> r.start;
        assertalways(is.get() == ':');
        is >> r.step;
        assertalways(is.get() == ':');
        is >> r.stop;

        return is;
    }

    bool operator==(const range& other) const
    {
        return this->start == other.start && this->step == other.step &&
               this->stop == other.stop &&
               this->step_method == other.step_method;
    }

    bool operator!=(const range& other) const { return !(*this == other); }
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
        auto itit = iterators.rbegin();
        auto rgit = --ranges.end();
        for (; itit != iterators.rend(); --rgit, ++itit) {
            ++*itit;

            if (*itit >= rgit->end()) {
                // we reached the end of iterators(i), we need to set this to
                // begin() and go up to iterators(i-1) in order to increment
                // this.
                *itit = rgit->begin();
            } else {
                return *this; // we're done, as incrementing iterators(i) did
                              // not cause us to reach ranges(i).end().
            }
        }

        // if we reached this point, it means that we have reached the end of
        // ranges(0), and hence we have tried every combo. At this point
        // iterators(i) = ranges(i).begin() for every i We need to set
        // iterators(i) = ranges(i).end() for every i to signal an end to the
        // iterations.
        itit = iterators.rbegin();
        rgit = --ranges.end();
        for (; itit != iterators.rend(); --rgit, ++itit) {
            *itit = rgit->end();
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

    /*! \brief Return number of combinations represented by the \name
     * multi_range_iterator object.
     */
    int count()
    {
        int cnt = 1;
        for (int i = 0; i < this->ranges.size(); i++)
            cnt *= this->ranges(i).count();
        return cnt;
    }
};

} // end namespace libbase