#ifndef __commsys_simulator_h
#define __commsys_simulator_h

#include "config.h"
#include "commsys_errorrates.h"
#include "experiment.h"
#include "randgen.h"
#include "commsys.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   Common Base for Communication Systems Simulator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

template <class S, class R=commsys_errorrates>
class basic_commsys_simulator
   : public experiment_binomial, public R {
protected:
   /*! \name Bound objects */
   //! Flag to indicate whether the objects should be released on destruction
   bool  internallyallocated;
   libbase::randgen     *src;    //!< Source data sequence generator
   commsys<S>           *sys;    //!< Communication systems
   // @}
   /*! \name Internal state */
   libbase::vector<int> last_event;
   // @}
protected:
   /*! \name Setup functions */
   void init();
   void clear();
   void free();
   // @}
   /*! \name Internal functions */
   libbase::vector<int> createsource();
   void cycleonce(libbase::vector<double>& result);
   // @}
   // System Interface for Results
   int get_iter() const { return sys->getcodec()->num_iter(); };
   int get_symbolsperblock() const { return sys->getcodec()->block_size() - sys->getcodec()->tail_length(); };
   int get_alphabetsize() const { return sys->getcodec()->num_inputs(); };
   int get_bitspersymbol() const { return int(round(log2(double(get_alphabetsize())))); };
public:
   /*! \name Constructors / Destructors */
   basic_commsys_simulator(libbase::randgen *src, commsys<S> *sys);
   basic_commsys_simulator(const basic_commsys_simulator<S,R>& c);
   basic_commsys_simulator() { clear(); };
   virtual ~basic_commsys_simulator() { free(); };
   // @}

   // Experiment parameter handling
   void seedfrom(libbase::random& r);
   void set_parameter(const double x) { sys->getchan()->set_parameter(x); };
   double get_parameter() const { return sys->getchan()->get_parameter(); };

   // Experiment handling
   void sample(libbase::vector<double>& result);
   int count() const { return R::count(); };
   int get_multiplicity(int i) const { return R::get_multiplicity(i); };
   std::string result_description(int i) const { return R::result_description(i); };
   libbase::vector<int> get_event() const { return last_event; };

   /*! \name Component object handles */
   //! Get communication system
   const commsys<S> *getsystem() const { return sys; };
   // @}

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

/*!
   \brief   Communication System Simulator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/
template <class S, class R=commsys_errorrates>
class commsys_simulator : public basic_commsys_simulator<S,R> {
public:
   // Serialization Support
   DECLARE_SERIALIZER(commsys_simulator)
};

}; // end namespace

#endif
