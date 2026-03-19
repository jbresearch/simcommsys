/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
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

#ifndef __experiment_h
#define __experiment_h

#include "config.h"
#include "parametric.h"
#include "queryable.h"
#include "random.h"
#include "serializer.h"
#include "vector.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace libcomm
{

/*!
 * \brief   Generic experiment.
 * \author  Johann Briffa
 */

class experiment : public parametric,
                   public queryable,
                   public libbase::serializable
{
private:
    /*! \name Internal variables */
    uint64_t samplecount; //!< Number of samples accumulated
                          // @}

protected:
    /*! \name Helpers for derived classes */
    /*!
     * \brief Add 'b' to 'a', initializing 'a' if necessary
     * \param[in,out] a Accumulator vector
     * \param[in] b Vector to be added to accumulator
     */
    template <class T>
    static void safe_accumulate(libbase::vector<T>& a,
                                const libbase::vector<T>& b)
    {
        if (a.size() == 0) {
            a = b;
        } else {
            a += b;
        }
    }
    // @}

    /*! \name Result accumulator interface */
    /*!
     * \brief Reset accumulated results
     */
    virtual void derived_reset() = 0;
    /*!
     * \brief Add the given sample results to the accumulated set
     * \param[in] sample_result   Vector containing a set of results
     * \param[in] sample_count    Vector containing a set of counts
     */
    virtual void derived_accumulate_result(
        const libbase::vector<double>& sample_result,
        const libbase::vector<uint64_t>& sample_count) = 0;
    /*!
     * \brief Add the complete state of results to the accumulated set
     * \param[in] state_values Vector set of accumulated results
     * \param[in] state_counts Vector set of accumulated counts
     */
    virtual void
    derived_accumulate_state(const libbase::vector<double>& state_values,
                             const libbase::vector<uint64_t>& state_counts) = 0;
    // @}

public:
    /*! \name Constructors / Destructors */
    virtual ~experiment() {}
    // @}

    /*! \name Experiment parameter handling */
    //! Seeds any random generators from a pseudo-random sequence
    virtual void seedfrom(libbase::random& r) = 0;
    // @}

    /*! \name Experiment handling */
    /*!
     * \brief Perform the experiment and return a single sample
     * \param[out] sample_result   The set of results for the experiment
     * \param[out] sample_count    The set of counts for the experiment
     */
    virtual void sample(libbase::vector<double>& sample_result,
                        libbase::vector<uint64_t>& sample_count) = 0;
    /*!
     * \brief The number of elements making up a sample
     * This getter is required by the results file writer, when writing the
     * header, as a result vector is not yet available at that point. Otherwise,
     * the value may be easily obtained from the size of the result in sample().
     */
    virtual int result_count() const = 0;
    /*!
     * \brief Title/description of result at index 'i'
     */
    virtual std::string result_description(int i) const = 0;
    /*!
     * \brief Return the simulated event from the last sample
     * \return An experiment-specific description of the last event
     *
     * This hook is used by the showerrorevent program, which assumes the
     * vector contains a concatenation of the source and decoded vectors
     * for the current frame.
     */
    virtual libbase::vector<int> get_event() const = 0;
    /*!
     * \brief Get the complete state of accumulated results
     * \param[out] state_values Vector set of accumulated results
     * \param[out] state_counts Vector set of accumulated counts
     */
    virtual void get_state(libbase::vector<double>& state_values,
                           libbase::vector<uint64_t>& state_counts) const = 0;
    /*!
     * \brief Determine result estimate based on accumulated set
     * \param[out] estimate Vector containing the set of estimates
     * \param[out] stderror Vector containing the corresponding standard error
     */
    virtual void estimate(libbase::vector<double>& estimate,
                          libbase::vector<double>& stderror) const = 0;
    // @}

    /*! \name Result accumulator interface */
    /*!
     * \brief Reset accumulated results
     */
    void reset()
    {
        samplecount = 0;
        derived_reset();
    }
    /*!
     * \brief Add the given sample results to the accumulated set
     * \param[in] sample_result   Vector containing a set of results
     * \param[in] sample_count    Vector containing a set of counts
     */
    void accumulate_result(const libbase::vector<double>& sample_result,
                           const libbase::vector<uint64_t>& sample_count)
    {
        samplecount++;
        derived_accumulate_result(sample_result, sample_count);
    }
    /*!
     * \brief Add the complete state of results to the accumulated set
     * \param[in] samplecount The number of samples in the accumulated set
     * \param[in] state_values Vector set of accumulated results
     * \param[in] state_counts Vector set of accumulated counts
     */
    void accumulate_state(uint64_t samplecount,
                          const libbase::vector<double>& state_values,
                             const libbase::vector<uint64_t>& state_counts)
    {
        this->samplecount += samplecount;
        derived_accumulate_state(state_values, state_counts);
    }
    /*!
     * \brief The number of samples taken to produce the result
     */
    uint64_t get_samplecount() const { return samplecount; }
    /*!
     * \brief Display accumulated results in human-readable form
     */
    void prettyprint_results(std::ostream& sout,
                             const libbase::vector<double>& result,
                             const libbase::vector<double>& errormargin) const;
    // @}

    /*! \name Description */
    //! Human-readable experiment description
    virtual std::string description() const = 0;
    // @}

    // Serialization Support
    DECLARE_BASE_SERIALIZER(experiment)
};

} // namespace libcomm

#endif
