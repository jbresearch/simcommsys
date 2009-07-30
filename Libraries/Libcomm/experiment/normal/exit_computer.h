#ifndef __exit_computer_h
#define __exit_computer_h

#include "config.h"
#include "experiment/experiment_normal.h"
#include "commsys.h"
#include "randgen.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   EXIT Chart Computer.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * \warning Currently this is just a placeholder; functionality still needs
 * to be written.
 */

template <class S>
class exit_computer : public experiment_normal {
protected:
   /*! \name Bound objects */
   //! Flag to indicate whether the objects should be released on destruction
   bool internallyallocated;
   libbase::randgen *src; //!< Source data sequence generator
   commsys<S> *sys; //!< Communication systems
   // @}
   /*! \name Internal state */
   libbase::vector<int> last_event;
   // @}
protected:
   /*! \name Setup functions */
   void clear();
   void free();
   // @}
   /*! \name Internal functions */
   libbase::vector<int> createsource();
   void cycleonce(libbase::vector<double>& result);
   // @}
public:
   /*! \name Constructors / Destructors */
   exit_computer(libbase::randgen *src, commsys<S> *sys);
   exit_computer(const exit_computer<S>& c);
   exit_computer()
      {
      clear();
      }
   virtual ~exit_computer()
      {
      free();
      }
   // @}

   // Experiment parameter handling
   void seedfrom(libbase::random& r);
   void set_parameter(const double x)
      {
      sys->getchan()->set_parameter(x);
      }
   double get_parameter() const
      {
      return sys->getchan()->get_parameter();
      }

   // Experiment handling
   void sample(libbase::vector<double>& result);
   int count() const
      {
      return 1;
      }
   int get_multiplicity(int i) const
      {
      return 1;
      }
   std::string result_description(int i) const
      {
      return "";
      }
   libbase::vector<int> get_event() const
      {
      return last_event;
      }

   /*! \name Component object handles */
   //! Get communication system
   const commsys<S> *getsystem() const
      {
      return sys;
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(exit_computer);
};

} // end namespace

#endif
