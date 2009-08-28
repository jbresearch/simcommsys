#ifndef __experiment_binomial_h
#define __experiment_binomial_h

#include "experiment.h"

namespace libcomm {

/*!
 * \brief   Experiment for estimation of a binomial proportion.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * Implements the accumulator functions required by the experiment class.
 */

class experiment_binomial : public experiment {
   /*! \name Internal variables */
   libbase::vector<double> sum; //!< Vector of result sums
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
   void estimate(libbase::vector<double>& estimate,
         libbase::vector<double>& stderror) const;
};

} // end namespace

#endif
