#ifndef __filter_h
#define __filter_h

#include "config.h"
#include "matrix.h"

namespace libimage {

/*
 * \brief   Filter interface
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This class specifies the interface that any filter class should provide.
 * The specification supports tiled filtering through a two-pass process.
 * The first pass gathers details from the image, tile by tile, while the
 * second pass uses the gathered information for any parameter estimates
 * (such as automatic thresholds, etc) and applies the filter to the image.
 */

template <class T>
class filter {
public:
   virtual ~filter()
      {
      }
   // progress display
   virtual void display_progress(const int done, const int total) const = 0;
   // parameter estimation (updates internal statistics)
   virtual void reset() = 0;
   virtual void update(const libbase::matrix<T>& in) = 0;
   virtual void estimate() = 0;
   //! Filter process loop (only updates output matrix)
   virtual void
   process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const = 0;
   //! Apply filter to an image channel
   void apply(const libbase::matrix<T>& in, libbase::matrix<T>& out);
};



} // end namespace

#endif
