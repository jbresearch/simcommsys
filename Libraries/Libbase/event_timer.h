#ifndef __event_timer_h
#define __event_timer_h

#include "config.h"
#include "timer.h"
#include "rvstatistics.h"

namespace libbase {

/*!
 * \brief   Event time-keeper.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This class is used to gather statistics on events, automatically printing
 * these statistics when the object goes out of scope. This makes it easy to
 * print data over multiple calls of a method, by creating a static object
 * within the method itself.
 *
 * The class takes one template argument, for the type of timer to use. This
 * allows the use of CPU or other embedded timers (such as GPU timers). The
 * timer must provide start(), stop(), and elapsed() methods.
 */

template <class Timer>
class event_timer {
private:
   std::string name; //!< Event name, for user to identify statistics
   Timer t;
   libbase::rvstatistics stat;
public:
   // constructor
   explicit event_timer(const std::string& name) :
      name(name), t(name)
      {
      }
   // destructor, prints collected statistics
   ~event_timer()
      {
      std::cerr << name << ": " << stat.count() << " events, total = "
            << libbase::timer::format(stat.sum()) << ", mean = "
            << libbase::timer::format(stat.mean()) << ", std = "
            << libbase::timer::format(stat.sigma()) << std::endl;
      }
   /* \name Event collection */
   //! Called at the beginning of an event
   void start()
      {
      t.start();
      }
   //! Called at the completion of an event
   void stop()
      {
      t.stop();
      stat.insert(t.elapsed());
      }
};

} // end namespace

#endif
