#ifndef __filter_h
#define __filter_h

#include "config.h"
#include "vcs.h"
#include "matrix.h"

/*
  Version 1.00 (15 Apr 2002)
  Initial version. This class is a base class for filters; it specifies the interface
  that any filter class should provide. The specification supports tiled filtering
  through a two-pass process. The first pass gathers details from the image, tile by
  tile, while the second pass uses the gathered information for any parameter estimates
  (such as automatic thresholds, etc) and applies the filter to the image.

  Version 1.10 (29 Apr 2002)
  added hook for progress display.

  Version 1.20 (10 Nov 2006)
  * defined class and associated data within "libimage" namespace.
*/

namespace libimage {

extern const libbase::vcs filter_version;

template <class T> class filter {
public:
   virtual ~filter() {};
   // progress display
   virtual void display_progress(const int done, const int total) const = 0;
   // parameter estimation (updates internal statistics)
   virtual void reset() = 0;
   virtual void update(const libbase::matrix<T>& in) = 0;
   virtual void estimate() = 0;
   // filter process loop (only updates output matrix)
   virtual void process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const = 0;
};

}; // end namespace

#endif
