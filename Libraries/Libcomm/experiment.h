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
#include "serializer.h"
#include "vector.h"
#include "random.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
 * \brief   Generic experiment.
 * \author  Johann Briffa
 */

class experiment : public parametric, public libbase::serializable {
private:
   /*! \name Internal variables */
   libbase::int64u samplecount; //!< Number of samples accumulated
   // @}

protected:
   /*! \name Helpers for derived classes */
   /*!
    * \brief Add 'b' to 'a', initializing 'a' if necessary
    * \param[in,out] a Accumulator vector
    * \param[in] b Vector to be added to accumulator
    */
   static void safe_accumulate(libbase::vector<double>& a,
         const libbase::vector<double>& b)
      {
      if (a.size() == 0)
         a = b;
      else
         a += b;
      }
   // @}

   /*! \name Result accumulator interface */
   /*!
    * \brief Reset accumulated results
    */
   virtual void derived_reset() = 0;
   /*!
    * \brief Add the given sample results to the accumulated set
    * \param[in] result   Vector containing a set of results
    */
   virtual void derived_accumulate(const libbase::vector<double>& result) = 0;
   /*!
    * \brief Add the complete state of results to the accumulated set
    * \param[in] state Vector set of accumulated results
    */
   virtual void accumulate_state(const libbase::vector<double>& state) = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   virtual ~experiment()
      {
      }
   // @}

   /*! \name Experiment parameter handling */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r) = 0;
   // @}

   /*! \name Experiment handling */
   /*!
    * \brief Perform the experiment and return a single sample
    * \param[out] result   The set of results for the experiment
    */
   virtual void sample(libbase::vector<double>& result) = 0;
   /*!
    * \brief The number of elements making up a sample
    * \note This getter is likely to be redundant, as the value may be
    * easily obtained from the size of result in sample()
    * \todo Remove this method from interface.
    */
   virtual int count() const = 0;
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
    * \param[in] result   Vector containing a set of results
    */
   void accumulate(const libbase::vector<double>& result)
      {
      samplecount++;
      derived_accumulate(result);
      }
   /*!
    * \brief Add the complete state of results to the accumulated set
    * \param[in] samplecount The number of samples in the accumulated set
    * \param[in] state Vector set of accumulated results
    */
   void accumulate_state(libbase::int64u samplecount, const libbase::vector<
         double>& state)
      {
      this->samplecount += samplecount;
      accumulate_state(state);
      }
   /*!
    * \brief Get the complete state of accumulated results
    * \param[out] state Vector set of accumulated results
    */
   virtual void get_state(libbase::vector<double>& state) const = 0;
   /*!
    * \brief Determine result estimate based on accumulated set
    * \param[out] estimate Vector containing the set of estimates
    * \param[out] stderror Vector containing the corresponding standard error
    */
   virtual void estimate(libbase::vector<double>& estimate, libbase::vector<
         double>& stderror) const = 0;
   /*!
    * \brief The number of samples taken to produce the result
    */
   libbase::int64u get_samplecount() const
      {
      return samplecount;
      }
   /*!
    * \brief The number of samples taken to produce result 'i'
    */
   libbase::int64u get_samplecount(int i) const
      {
      return get_samplecount() * get_multiplicity(i);
      }
   /*!
    * \brief The number of elements/sample for result 'i'
    * \param[in]  i  Result index
    * \return        Population size per sample for given result index
    */
   virtual int get_multiplicity(int i) const = 0;
   /*!
    * \brief Display accumulated results in human-readable form
    */
   virtual void prettyprint_results(std::ostream& sout, const libbase::vector<
         double>& result, const libbase::vector<double>& errormargin) const;
   // @}

   /*! \name Description */
   //! Human-readable experiment description
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(experiment)
};

} // end namespace

#endif
