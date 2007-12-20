#ifndef __bstream_h
#define __bstream_h

#include "config.h"
#include "bitfield.h"

namespace libbase {

/*!
   \brief   Bitstream Base.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

   \version 1.02 (15 Jun 2002)
   - modified implementation file min() function usage to specify global one.
   - added default constructor and virtual destructor for obstream/ibstream

   \version 1.10 (26 Oct 2006)
   - defined class and associated data within "libbase" namespace.
*/

class bstream {
protected:
   bitfield     buffer; // a 32-bit buffer for read/write operations
   int          ptr;            // points to the first unused/unread bit
public:
   bstream();
};

class obstream : public bstream {
protected:
   virtual void write_buffer() = 0;
public:
   obstream() {};
   virtual ~obstream() {};
   obstream& operator<<(const bitfield& b);
};

class ibstream : public bstream {
protected:
   virtual void read_buffer() = 0;
public:
   ibstream() {};
   virtual ~ibstream() {};
   ibstream& operator>>(bitfield& b);
};

}; // end namespace

#endif
