#ifndef __experiment_normal_h
#define __experiment_normal_h

#include "experiment.h"

namespace libcomm {

/*!
 * \brief   Experiment with normally distributed samples.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * Implements the accumulator functions required by the experiment class,
 * moved from previous implementation in montecarlo.
 */

class experiment_normal : public experiment {
   /*! \name Internal variables */
   libbase::vector<double> sum; //!< Vector of result sums
   libbase::vector<double> sumsq; //!< Vector of result sum-of-squares
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
