#ifndef __bstream_h
#define __bstream_h

#include "config.h"
#include "vcs.h"
#include "bitfield.h"

extern const vcs bstream_version;

class bstream {
protected:
   bitfield	buffer;	// a 32-bit buffer for read/write operations
   int		ptr;		// points to the first unused/unread bit
public:
   bstream();
};

class obstream : public bstream {
protected:
   virtual void write_buffer() = 0;
public:
   obstream& operator<<(const bitfield& b);
};

class ibstream : public bstream {
protected:
   virtual void read_buffer() = 0;
public:
   ibstream& operator>>(bitfield& b);
};

#endif
