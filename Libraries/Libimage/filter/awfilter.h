#ifndef __awfilter_h
#define __awfilter_h

#include "filter.h"
#include "rvstatistics.h"

namespace libimage {

/*
 * \brief   Adaptive Wiener Filter
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This filter implements Lee's algorithm (Lee, 1980), as used in Matlab's
 * wiener2 filter function (in the image processing library).
 * Note that Matlab provides a way for the function to estimate the noise
 * variance itself - this is actually computed as the mean value of the image
 * local variance. This class allows this to be done by using the appropriate
 * constructor. The estimator function is also publicly available.
 */

template <class T>
class awfilter : public filter<T> {
protected:
   // user-supplied settings
   int m_d; //!< greatest distance from current pixel in neighbourhood
   bool m_autoestimate; //!< flag for automatic estimation of noise energy
   double m_noise; //!< estimate of noise energy to remove
   // internal variables
   libbase::rvstatistics rvglobal;
public:
   awfilter(const int d, const double noise)
      {
      init(d, noise);
      }
   awfilter(const int d)
      {
      init(d);
      }
   // initialization
   void init(const int d, const double noise);
   void init(const int d);
   // progress display
   void display_progress(const int done, const int total) const
      {
      }
   // parameter estimation (updates internal statistics)
   void reset();
   void update(const libbase::matrix<T>& in);
   void estimate();
   double get_estimate() const
      {
      return m_noise;
      }
   // filter process loop (only updates output matrix)
   void process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const;
};

} // end namespace

#endif
