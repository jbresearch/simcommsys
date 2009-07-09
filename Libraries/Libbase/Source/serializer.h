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

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 Class created to allow the implementation of serialization functions
 similar to MFC's. Concept is that derived classes should have a static
 const serializer object within the class, which should be constructed
 by giving it:
 - the name of the derived class,
 - the name of its base class (to which we would like to add support),
 - and a pointer to a function which creates a new object of the derived
 type. This would ideally be a static function for the derived class.
 The base class then has access to these functions through a static
 serializer function, to which it needs to give:
 - its own name (ie that of the base class),
 - and the derived class's name, the one to which it needs access.
 This can be used within an istream >> function to dynamically create
 & load a derived class object.

 In serializable classes, the stream << output function first writes the
 name of the derived class, then calls its serialize() to output the data.
 The name is obtained from the virtual name() function.
 Similarly, the stream >> input function first gets the name from the stream,
 then (via serialize::call) creates a new object of the appropriate type and
 calls its serialize() function to get the relevant data.

 \note In the constructor, the function pointer is passed directly, not by
 reference. This is required to pass anything except global functions.

 \note The map is held as a static pointer to map. For some obscure reason,
 static or global maps are leading to access violations. The pointer is
 initially NULL, and when the first serializer object is created, space
 for this is allocated.
 A static counter keeps track of how many serializer objects are
 defined. When this drops to zero, the global map is deallocated. Note
 that this should only drop to zero on program end, assuming all
 serializer objects are created with either global or static member
 scope.

 \note Macros are defined to standardize declarations in serializable
 classes; this mirrors what Microsoft do in MFC.
 */

class serializer {
public:
   typedef void*(*fptr)();
private:
   static std::map<std::string, fptr>* cmap;
   static int count;
   std::string classname;
public:
   static void* call(const std::string& base, const std::string& derived);
public:
   serializer(const std::string& base, const std::string& derived, fptr func);
   ~serializer();
   const char *name() const
      {
      return classname.c_str();
      }
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
   friend std::ostream& operator<<(std::ostream& sout, const class_name* x) \
      { \
      sout << x->name() << "\n"; \
      x->serialize(sout); \
      return sout; \
      }; \
   /*! \brief Stream input */ \
   friend std::istream& operator>>(std::istream& sin, class_name*& x) \
      { \
      std::string name; \
      std::streampos start = sin.tellg(); \
      sin >> name; \
      x = (class_name*) libbase::serializer::call(#class_name, name); \
      if(x == NULL) \
         { \
         sin.seekg( start ); \
         sin.clear( std::ios::failbit ); \
         } \
      else \
         assertalways(x->serialize(sin)); \
      return sin; \
      } \
   /* @} */

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
   std::istream& serialize(std::istream& sin)

#define DECLARE_CONCRETE_BASE_SERIALIZER( class_name ) \
   private: \
   /*! \name Serialization Support */ \
   /*! \brief Serialization helper object */ \
   static const libbase::serializer shelper; \
   /*! \brief Heap creation function */ \
   static void* create() { return new class_name; }; \
   /* @} */ \
   public: \
   /*! \name Serialization Support */ \
   /*! \brief Cloning operation */ \
   virtual class_name *clone() const { return new class_name(*this); }; \
   /*! \brief Derived object's name */ \
   virtual const char* name() const { return shelper.name(); }; \
   /*! \brief Serialization output */ \
   virtual std::ostream& serialize(std::ostream& sout) const; \
   /*! \brief Serialization input */ \
   virtual std::istream& serialize(std::istream& sin); \
   /*! \brief Stream output */ \
   friend std::ostream& operator<<(std::ostream& sout, const class_name* x) \
      { \
      sout << x->name() << "\n"; \
      x->serialize(sout); \
      return sout; \
      }; \
   /*! \brief Stream input */ \
   friend std::istream& operator>>(std::istream& sin, class_name*& x) \
      { \
      std::string name; \
      std::streampos start = sin.tellg(); \
      sin >> name; \
      x = (class_name*) libbase::serializer::call(#class_name, name); \
      if(x == NULL) \
         { \
         sin.seekg( start ); \
         sin.clear( std::ios::failbit ); \
         } \
      else \
         assertalways(x->serialize(sin)); \
      return sin; \
      } \
   /* @} */

} // end namespace

#endif
