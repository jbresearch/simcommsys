#ifndef __serializer_h
#define __serializer_h

#include "config.h"
#include <map>
#include <string>
#include <iostream>

namespace libbase {

/*!
   \brief   Serialization helper.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (28 Feb -> 6 Mar 2002)
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

   \version 1.01 (7 Mar 2002)
   modified the constructor so that the createfunc is passed directly, not by reference.
   This is required to pass anything except global functions I think.

   \version 1.02 (8 Mar 2002)
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

   \version 1.03 (8 Mar 2002)
   added a public function which returns the name of the class, as given to the constructor
   (given as the derived class name). This is intended to be used in any name-returning
   functions within the derived class to ensure consistency.

   \version 1.10 (11 Mar 2002)
   changed the definition of createfunc - the function now takes no parameters, so that
   it should only create an object of the required type; a stream loading needs to be
   called separately. Also, cmap now maps the string to a createfunc object rather than
   and integer. This ensures compatibility with architectures where a function pointer
   and an integer have different sizes.

   \version 1.11 (24 Mar 2002)
   added debugging tracers in creation and call functions.

   \version 1.20 (26 Oct 2006)
   - defined class and associated data within "libbase" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.21 (24 Apr 2007)
   - moved typedef createfunc to be defined as a public element of the class
   - refactored, renaming it to fptr

   \version 1.22 (22 Jan 2008)
   - Changed debug output to go to trace instead of clog.

   \version 2.00 (24 Apr 2008)
   - Created macros to standardize functions declarations in serializable classes;
     this mirrors what Microsoft do in MFC.
   - Added inclusion of iostream in this class, so derived classes shall
     not need it
*/

class serializer {
public:
   typedef void*(*fptr)();
private:
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

#define DECLARE_BASE_SERIALIZER( class_name ) \
   public: \
   /*! \name Serialization Support */ \
   /*! \brief Cloning operation */ \
   virtual class_name *clone() const = 0; \
   /*! \brief Derived object's name */ \
   virtual const char* name() const = 0; \
   /*! \brief Serialization output */ \
   virtual std::ostream& serialize(std::ostream& sout) const = 0; \
   /*! \brief Serialization input */ \
   virtual std::istream& serialize(std::istream& sin) = 0; \
   /*! \brief Stream output */ \
   friend std::ostream& operator<<(std::ostream& sout, const class_name* x); \
   /*! \brief Stream input */ \
   friend std::istream& operator>>(std::istream& sin, class_name*& x); \
   /* @} */

#define IMPLEMENT_BASE_SERIALIZER( class_name ) \
   std::ostream& operator<<(std::ostream& sout, const class_name* x) \
      { \
      sout << x->name() << "\n"; \
      x->serialize(sout); \
      return sout; \
      } \
   std::istream& operator>>(std::istream& sin, class_name*& x) \
      { \
      std::string name; \
      sin >> name; \
      x = (class_name*) libbase::serializer::call(#class_name, name); \
      if(x == NULL) \
         { \
         std::cerr << "FATAL ERROR (" #class_name "): Type \"" << name << "\" unknown.\n"; \
         exit(1); \
         } \
      x->serialize(sin); \
      return sin; \
      }

#define DECLARE_SERIALIZER( class_name ) \
   private: \
   /*! \name Serialization Support */ \
   /*! \brief Serialization helper object */ \
   static const libbase::serializer shelper; \
   /*! \brief Heap creation function */ \
   static void* create() { return new class_name; }; \
   /* @} */ \
   public: \
   class_name *clone() const { return new class_name(*this); }; \
   const char* name() const { return shelper.name(); }; \
   std::ostream& serialize(std::ostream& sout) const; \
   std::istream& serialize(std::istream& sin);

}; // end namespace

#endif
