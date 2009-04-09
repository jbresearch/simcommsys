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
   \brief   Generic experiment.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

class experiment : public parametric {
private:
   /*! \name Internal variables */
   libbase::int64u   samplecount;               //!< Number of samples accumulated
   // @}

protected:
   /*! \name Result accumulator interface */
   /*!
      \brief Reset accumulated results
   */
   virtual void derived_reset() = 0;
   /*!
      \brief Add the given sample results to the accumulated set
      \param[in] result   Vector containing a set of results
   */
   virtual void derived_accumulate(const libbase::vector<double>& result) = 0;
   /*!
      \brief Add the complete state of results to the accumulated set
      \param[in] state Vector set of accumulated results
   */
   virtual void accumulate_state(const libbase::vector<double>& state) = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   virtual ~experiment() {};
   // @}

   /*! \name Experiment parameter handling */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r) = 0;
   // @}

   /*! \name Experiment handling */
   /*!
      \brief Perform the experiment and return a single sample
      \param[out] result   The set of results for the experiment
   */
   virtual void sample(libbase::vector<double>& result) = 0;
   /*!
      \brief The number of elements making up a sample
      \note This getter is likely to be redundant, as the value may be
            easily obtained from the size of result in sample()
      \callergraph
   */
   virtual int count() const = 0;
   /*!
      \brief Title/description of result at index 'i'
   */
   virtual std::string result_description(int i) const = 0;
   /*!
      \brief Return the simulated event from the last sample
      \return An experiment-specific description of the last event
   */
   virtual libbase::vector<int> get_event() const = 0;
   // @}

   /*! \name Result accumulator interface */
   /*!
      \brief Reset accumulated results
   */
   void reset() { samplecount = 0; derived_reset(); };
   /*!
      \brief Add the given sample results to the accumulated set
      \param[in] result   Vector containing a set of results
   */
   void accumulate(const libbase::vector<double>& result)
      { samplecount++; derived_accumulate(result); };
   /*!
      \brief Add the complete state of results to the accumulated set
      \param[in] samplecount The number of samples in the accumulated set
      \param[in] state Vector set of accumulated results
   */
   void accumulate_state(libbase::int64u samplecount, const libbase::vector<double>& state)
      { this->samplecount += samplecount; accumulate_state(state); };
   /*!
      \brief Get the complete state of accumulated results
      \param[out] state Vector set of accumulated results
   */
   virtual void get_state(libbase::vector<double>& state) const = 0;
   /*!
      \brief Determine result estimate based on accumulated set
      \param[out] estimate Vector containing the set of estimates
      \param[out] stderror Vector containing the corresponding standard error
   */
   virtual void estimate(libbase::vector<double>& estimate, libbase::vector<double>& stderror) const = 0;
   /*!
      \brief The number of samples taken to produce the result
   */
   libbase::int64u get_samplecount() const { return samplecount; };
   /*!
      \brief The number of samples taken to produce result 'i'
   */
   libbase::int64u get_samplecount(int i) const { return get_samplecount() * get_multiplicity(i); };
   /*!
      \brief The number of elements/sample for result 'i'
      \param[in]  i  Result index
      \return        Population size per sample for given result index
   */
   virtual int get_multiplicity(int i) const = 0;
   // @}

   /*! \name Description */
   //! Human-readable experiment description
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(experiment);
};

}; // end namespace

#endif
