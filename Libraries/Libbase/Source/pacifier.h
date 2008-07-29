#ifndef __pacifier_h
#define __pacifier_h

#include "config.h"
#include "timer.h"
#include <string>

namespace libbase {

/*!
   \brief   User Pacifier.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   A class that formats output suitable for keeping user updated with the
   progress of an operation.
*/

class pacifier {
private:
   static bool quiet;
public:
   /*! \name Static interface */
   static void enable_output() { quiet = false; };
   static void disable_output() { quiet = true; };
   // @}

private:
   std::string name;
   timer       t;
   int         last;
   size_t      characters;
public:
   /*! \name Constructors / Destructors */
   explicit pacifier(const std::string& name="Process");
   virtual ~pacifier();
   // @}

   /*! \name Pacifier operation */
   std::string update(int complete, int total=100);
   // @}
};

}; // end namespace

#endif
