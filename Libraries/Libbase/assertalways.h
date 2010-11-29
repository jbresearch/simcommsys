#ifndef __assert_h
#define __assert_h

/*!
 * \file
 * \brief   Main Configuration.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

// system include files

#include <cassert>

// *** Global namespace ***

// An assertion that is implemented even in release builds

#ifdef NDEBUG
#  define assertalways(_Expression) (void)( (!!(_Expression)) || (libbase::reportassertionandfail(#_Expression, __FILE__, __LINE__), 0) )
#else
#  define assertalways(_Expression) assert(_Expression)
#endif

// Fail with error

#define failwith(_String) libbase::reporterrorandfail(_String, __FILE__, __LINE__)

// *** Within library namespace ***

namespace libbase {

// Debugging tools

void reportassertionandfail(const char *expression, const char *file, int line);
void reporterrorandfail(const char *expression, const char *file, int line);

} // end namespace

#endif
