#ifndef __experiment_h
#define __experiment_h

#include "config.h"
#include "vector.h"
#include "serializer.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Generic experiment.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (2 Sep 1999)
   added a hook for clients to know the number of frames simulated in a particular run.

   \version 1.11 (26 Oct 2001)
   added a virtual destroy function (see interleaver.h)

   \version 1.12 (6 Mar 2002)
   changed vcs version variable from a global to a static class variable.

   \version 1.20 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.

   \version 1.30 (24 Apr 2007)
   - added serialization facility requirement, to facilitate passing the experiment
    over the network pipe for masterslave implementation.

   \version 1.40 (18 Dec 2007)
   - Modified definition of sample() so that only a *single* sample is performed.
     This is essential so that the moments of the results are meaningful and can
     be used to determine convergence/tolerance. Any multiple-sampling should be
     done elsewhere (e.g. in montecarlo::slave_work).

   \version 1.41 (17 Jan 2008)
   - Renamed set/get to set_parameter/get_parameter

   \version 1.42 (22 Jan 2008)
   - Removed 'friend' declaration of stream operators.

   \version 1.50 (6 May 2008)
   - replaced serialization support with macros
   - added function definitions for accumulating results and computing the
     related estimates and standard error. This moves the responsibility for
     doing this from montecarlo to the experiment object.
*/

class experiment {
private:
   /*! \name Internal variables */
   int   samplecount;               //!< Number of samples accumulated
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
   //! Reset function for random generator
   virtual void seed(int s) = 0;
   //! Set the system parameter at which we want to simulate
   virtual void set_parameter(double x) = 0;
   //! Get the system parameter at which we are simulating
   virtual double get_parameter() = 0;
   // @}

   /*! \name Experiment handling */
   /*!
      \brief Perform the experiment and return a single sample
      \param[out] result   Vector containing the set of results for the experiment
   */
   virtual void sample(libbase::vector<double>& result) = 0;
   /*!
      \brief The number of elements making up a sample
      \note This getter is likely to be redundant, as the value may be
            easily obtained from the size of result in sample()
      \callergraph
   */
   virtual int count() const = 0;
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
   void accumulate_state(const int samplecount, const libbase::vector<double>& state)
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
   int get_samplecount() const { return samplecount; };
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(experiment)
};

/*!
   \brief   Experiment with normally distributed samples.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (6 May 2008)
   - Initial version; implements the accumulator functions required by the
     experiment class, moved from current implementation in montecarlo.
*/

class experiment_normal : public experiment {
   /*! \name Internal variables */
   libbase::vector<double> sum;     //!< Vector of result sums
   libbase::vector<double> sumsq;   //!< Vector of result sum-of-squares
   // @}

protected:
   // Accumulator functions
   void derived_reset();
   void derived_accumulate(const libbase::vector<double>& result);
   void accumulate_state(const libbase::vector<double>& state);
   // @}

public:
   // Accumulator functions
   void get_state(libbase::vector<double>& state) const;
   void estimate(libbase::vector<double>& estimate, libbase::vector<double>& stderror) const;
};

/*!
   \brief   Experiment for estimation of a binomial proportion.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (6 May 2008)
   - Initial version; implements the accumulator functions required by the
     experiment class.
*/

class experiment_binomial : public experiment {
   /*! \name Internal variables */
   libbase::vector<double> sum;     //!< Vector of result sums
   // @}

protected:
   // Accumulator functions
   void derived_reset();
   void derived_accumulate(const libbase::vector<double>& result);
   void accumulate_state(const libbase::vector<double>& state);
   // @}

public:
   // Accumulator functions
   void get_state(libbase::vector<double>& state) const;
   void estimate(libbase::vector<double>& estimate, libbase::vector<double>& stderror) const;
};

}; // end namespace

#endif
