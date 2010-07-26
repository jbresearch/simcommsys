#ifndef __atmfilter_h
#define __atmfilter_h

#include "filter.h"

namespace libimage {

/*
 * \brief   Alpha-Trimmed Mean Filter
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This filter computes the alpha-trimmed mean within a given neighbourhood.
 */

template <class T>
class atmfilter : public filter<T> {
protected:
   int m_d; //!< greatest distance from current pixel in neighbourhood
   int m_alpha; //!< number of outliers to trim at each end before computing mean
public:
   atmfilter(const int d, const int alpha)
      {
      init(d, alpha);
      }
   // initialization
   void init(const int d, const int alpha);
   // progress display
   void display_progress(const int done, const int total) const
      {
      }
   // parameter estimation (updates internal statistics)
   void reset()
      {
      }
   void update(const libbase::matrix<T>& in)
      {
      }
   void estimate()
      {
      }
   // filter process loop (only updates output matrix)
   void process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const;
};

} // end namespace

#endif
