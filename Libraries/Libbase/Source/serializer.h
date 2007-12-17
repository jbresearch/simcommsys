#ifndef __serializer_h
#define __serializer_h

#include "config.h"
#include "vcs.h"

#include <map>
#include <string>

namespace libbase {

/*!
   \brief   Serialization helper.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.00 (28 Feb -> 6 Mar 2002)
  original version - created to allow the implementation of serialization functions
  similar to MFC's. Created the vcs version control variable within the class - if this
  works well, this concept will be promoted to all other classes.
  Concept is as follows: derived classes should have a static const serializer object
  within the class, which should be constructed by giving it the name of the derived
  class, the name of its base class (to which we would like to add support) and a pointer
  to a function which creates a new object of the derived type, and loads it from the
  supplied stream. The function mentioned would ideally be a static function for the 
  derived class. The base class then has access to these functions through a static
  serializer function, to which it needs to give its own name (ie that of the base
  class), and the derived class's name, the one to which it needs access. This can be
  used within an istream >> function to dynamically create & load a derived class
  object.

  Version 1.01 (7 Mar 2002)
  modified the constructor so that the createfunc is passed directly, not by reference.
  This is required to pass anything except global functions I think.

  Version 1.02 (8 Mar 2002)
  modified the static map to be a static pointer to map instead. For some obscure reason,
  static or global maps are leading to access violations. The pointer is initially NULL,
  and when the first serializer object is created, space for this is allocated. Also,
  the map now associates a string key (containing the base and derived class names) with
  an integer (which is actually the numerical value of a pointer to the createfunc
  function). This ensures the necessary default value of 0 at a cost of some static
  casts. Also added a static counter to keep track of how many serializer objects are
  defined. When this drops to zero, the global map is deallocated. Note that this should
  only drop to zero on program end, assuming all serializer objects are created with
  either global or static member scope.

  Version 1.03 (8 Mar 2002)
  added a public function which returns the name of the class, as given to the constructor
  (given as the derived class name). This is intended to be used in any name-returning
  functions within the derived class to ensure consistency.

  Version 1.10 (11 Mar 2002)
  changed the definition of createfunc - the function now takes no parameters, so that
  it should only create an object of the required type; a stream loading needs to be
  called separately. Also, cmap now maps the string to a createfunc object rather than
  and integer. This ensures compatibility with architectures where a function pointer
  and an integer have different sizes.

  Version 1.11 (24 Mar 2002)
  added debugging tracers in creation and call functions.

  Version 1.20 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
  
  Version 1.21 (24 Apr 2007)
  * moved typedef createfunc to be defined as a public element of the class
  * refactored, renaming it to fptr 
*/

class serializer {
public:
   typedef void*(*fptr)();
private:
   static const vcs version;
   static std::map<std::string,fptr>* cmap;
   static int count;
   std::string classname;
public:
   static void* call(const std::string& base, const std::string& derived);
public:
   serializer(const std::string& base, const std::string& derived, fptr func);
   ~serializer();
   const char *name() const { return classname.c_str(); };
};

}; // end namespace

#endif
