#ifndef __fbstream_h
#define __fbstream_h

#include "config.h"
#include "vcs.h"
#include "bstream.h"
#include <fstream.h>

extern const vcs fbstream_version;

class ofbstream : virtual public obstream {
   ofstream c;
   void write_buffer();
public:
   ofbstream(char name[]);
   ~ofbstream();
   void open(char name[]);
   void close();
   bool eof();
   bool fail();
   bool bad();
   bool good();
};

class ifbstream : virtual public ibstream {
   ifstream c;
   void read_buffer();
public:
   ifbstream(char name[]);
   ~ifbstream();
   void open(char name[]);
   void close();
   bool eof();
   bool fail();
   bool bad();
   bool good();
};


/*** inline functions ***/

inline bool ofbstream::eof()
   {
   return c.eof();
   }

inline bool ofbstream::fail()
   {
   return c.fail();
   }

inline bool ofbstream::bad()
   {
   return c.bad();
   }

inline bool ofbstream::good()
   {
   return c.good();
   }



inline bool ifbstream::eof()
   {
   return c.eof();
   }

inline bool ifbstream::fail()
   {
   return c.fail();
   }

inline bool ifbstream::bad()
   {
   return c.bad();
   }

inline bool ifbstream::good()
   {
   return c.good();
   }


#endif
