/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __serializer_h
#define __serializer_h

#include "config.h"
#include <map>
#include <list>
#include <string>
#include <iostream>
#include <typeinfo>

namespace libbase {

// Pre-definition

class serializable;

/*!
 * \brief   Serialization helper.
 * \author  Johann Briffa
 *
 * Class created to allow the implementation of serialization functions
 * similar to MFC's. Concept is that derived classes should have a static
 * const serializer object within the class, which should be constructed
 * by giving it:
 * - the name of the derived class,
 * - the name of its base class (to which we would like to add support),
 * - and a pointer to a function which creates a new object of the derived
 * type. This would ideally be a static function for the derived class.
 * The base class then has access to these functions through a static
 * serializer function, to which it needs to give:
 * - its own name (ie that of the base class),
 * - and the derived class's name, the one to which it needs access.
 * This can be used within an istream >> function to dynamically create
 * & load a derived class object.
 *
 * In serializable classes, the stream << output function first writes the
 * name of the derived class, then calls its serialize() to output the data.
 * The name is obtained from the virtual name() function.
 * Similarly, the stream >> input function first gets the name from the stream,
 * then (via serialize::call) creates a new object of the appropriate type and
 * calls its serialize() function to get the relevant data.
 *
 * \note In the constructor, the function pointer is passed directly, not by
 * reference. This is required to pass anything except global functions.
 *
 * \note The map is held as a static pointer to map; this is to avoid the
 * "static initialization order fiasco", where the map might be created after
 * the first call to the serializer constructor (from some global object).
 * This would lead to access violations / segfaults. The pointer is
 * initially NULL, and when the first serializer object is created, space
 * for this is allocated.
 * A static counter keeps track of how many serializer objects are
 * defined. When this drops to zero, the global map is deallocated. Note
 * that this should only drop to zero on program end, assuming all
 * serializer objects are created with either global or static member
 * scope.
 *
 * \note Macros are defined to standardize declarations in serializable
 * classes; this mirrors what Microsoft do in MFC.
 */

class serializer {
public:
   typedef serializable*(*fptr)();
private:
   static std::map<std::string, fptr>* cmap;
   static int count;
   std::string classname;
public:
   static serializable* call(const std::string& base,
         const std::string& derived);
   //! Returns a list of base classes
   static std::list<std::string> get_base_classes();
   //! Returns a list of derived classes for given base class
   static std::list<std::string> get_derived_classes(const std::string& base);
public:
   serializer(const std::string& base, const std::string& derived, fptr func);
   ~serializer();
   const char *name() const
      {
      return classname.c_str();
      }
};

/*!
 * \brief   Serializable class base.
 * \author  Johann Briffa
 *
 * All serializable classes inherit from this. Implements part of the required
 * functionality, in conjunction with DECLARE_BASE_SERIALIZER() macro.
 */

class serializable {
public:
   virtual ~serializable()
      {
      }
   /*! \name Serialization Support */
   /*! \brief Cloning operation */
   virtual serializable *clone() const = 0;
   /*! \brief Derived object's name */
   virtual const char* name() const = 0;
   /*! \brief Serialization output */
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   /*! \brief Serialization input */
   virtual std::istream& serialize(std::istream& sin) = 0;
   /* @} */
};

#define DECLARE_BASE_SERIALIZER( class_name ) \
   /* Comment */ \
   public: \
   /*! \brief Stream output */ \
   friend std::ostream& operator<<(std::ostream& sout, const class_name* x) \
      { \
      sout << x->name() << std::endl; \
      x->serialize(sout); \
      return sout; \
      } \
   /*! \brief Stream input */ \
   friend std::istream& operator>>(std::istream& sin, class_name*& x) \
      { \
      std::string name; \
      std::streampos start = sin.tellg(); \
      sin >> name; \
      x = dynamic_cast<class_name *> (libbase::serializer::call(#class_name, name)); \
      if(x == NULL) \
         { \
         sin.seekg( start ); \
         sin.clear( std::ios::failbit ); \
         } \
      else \
         { \
         x->serialize(sin); \
         libbase::verify(sin); \
         } \
      return sin; \
      } \
   /* @} */

#define DECLARE_SERIALIZER( class_name ) \
   /* Comment */ \
   private: \
   /*! \name Serialization Support */ \
   /*! \brief Serialization helper object */ \
   static const libbase::serializer shelper; \
   /*! \brief Heap creation function */ \
   static libbase::serializable* create() { return new class_name; } \
   /* @} */ \
   public: \
   libbase::serializable *clone() const { return new class_name(*this); } \
   const char* name() const { return shelper.name(); } \
   std::ostream& serialize(std::ostream& sout) const; \
   std::istream& serialize(std::istream& sin);

} // end namespace

#endif
