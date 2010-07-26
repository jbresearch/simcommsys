#ifndef __limitfilter_h
#define __limitfilter_h

#include "filter.h"

namespace libimage {

/*
 * \brief   Limit Filter
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This filter limits pixels between given values.
 */

template <class T>
class limitfilter : public filter<T> {
protected:
   T m_lo;
   T m_hi;
public:
   limitfilter()
      {
      }
   limitfilter(const T lo, const T hi)
      {
      init(lo, hi);
      }
   virtual ~limitfilter()
      {
      }
   // initialization
   void init(const T lo, const T hi);
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
   void process(libbase::matrix<T>& m) const;
};

} // end namespace

#endif
