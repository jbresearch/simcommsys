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

#ifndef __masterslave_h
#define __masterslave_h

#include "config.h"
#include "vector.h"
#include "socket.h"
#include "walltimer.h"
#include "cputimer.h"
#include "functor.h"
#include <map>

namespace libbase {

/*!
 * \brief   Socket-based Master-Slave computation.
 * \author  Johann Briffa
 *
 * Class supporting socket-based master-slave relationship.
 * - supports dynamic slave list
 *
 * \note RPC calls are now done by passing a string reference, which is used
 * as a key in a map list. Two new functions support this:
 * - fregister() allows registering of functions by derived classes, and
 * - fcall() to actually call them
 * Since this class cannot know the exact type of the function pointers,
 * these are held by functors.
 *
 * \todo Serialize to network byte order always.
 *
 * \todo Consider modifying cmpi to support this class interface model, and
 * create a new abstract class to encapsulate both models.
 *
 * \todo Make setting priority effective on Windows
 *
 * \todo Split master and slave classes
 */

class masterslave {
private:
   // Internally-used constants (tags)
   typedef enum {
      tag_getname = 0xFA, tag_getcputime, tag_work = 0xFE, tag_die
   } tag_t;

   // communication objects
public:
   // slave state
   typedef enum {
      state_new = 0, state_eventpending, state_idle, state_working
   } state_t;
   // operating mode - returned by enable()
   typedef enum {
      mode_local = 0, mode_master, mode_slave
   } mode_t;

   // items for use by everyone (?)
private:
   bool initialized;
   double cputimeused;
   walltimer twall;
   cputimer tcpu;
public:
   // global enable of cluster system
   mode_t enable(const std::string& endpoint, bool quiet = false, int priority =
         10);
   // informative functions
   bool isenabled() const
      {
      return initialized;
      }
   double getcputime() const
      {
      return initialized ? cputimeused : tcpu.elapsed();
      }
   double getwalltime() const
      {
      return twall.elapsed();
      }
   double getusage() const
      {
      return getcputime() / getwalltime();
      }
   //! Reset CPU usage accumulation
   void resetcputime()
      {
      cputimeused = 0;
      }

   // items for use by slaves
private:
   std::map<std::string, std::shared_ptr<functor> > fmap;
   std::shared_ptr<socket> master;
   // helper functions
   void close();
   void setpriority(const int priority);
   void connect(const std::string& hostname, const int16u port);
   std::string gethostname();
   int gettag();
   void sendname();
   void sendcputime();
   void dowork();
   void slaveprocess(const std::string& hostname, const int16u port,
         const int priority);
public:
   // RPC function calls
   //! Register a RPC function
   void fregister(const std::string& name, std::shared_ptr<functor> f);
   //! Call a RPC function
   void fcall(const std::string& name);
   // slave -> master communication
   void send(const void *buf, const size_t len);
   void send(const int x)
      {
      send(&x, sizeof(x));
      }
   void send(const int64u x)
      {
      send(&x, sizeof(x));
      }
   void send(const double x)
      {
      send(&x, sizeof(x));
      }
   void send(const vector<double>& x);
   void send(const std::string& x);
   void receive(void *buf, const size_t len);
   void receive(int& x)
      {
      receive(&x, sizeof(x));
      }
   void receive(int64u& x)
      {
      receive(&x, sizeof(x));
      }
   void receive(double& x)
      {
      receive(&x, sizeof(x));
      }
   void receive(std::string& x);

   // items for use by master
private:
   std::map<std::shared_ptr<socket>, state_t> smap;
   // helper functions
   void close(std::shared_ptr<socket> s);
public:
   // creation and destruction
   masterslave() :
         initialized(false), cputimeused(0), twall("masterslave-wall", false), tcpu(
               "masterslave-cpu", false)
      {
      }
   ~masterslave()
      {
      disable();
      }
   // disable process
   void disable();
   // slave-interface functions
   std::shared_ptr<socket> find_new_slave();
   std::shared_ptr<socket> find_idle_slave();
   std::shared_ptr<socket> find_pending_slave();
   int count_workingslaves() const;
   bool anyoneworking() const;
   void waitforevent(const bool acceptnew = true, const double timeout = 0);
   void resetslave(std::shared_ptr<socket> s);
   void resetslaves();
   // informative functions
   size_t getnumslaves() const
      {
      return smap.size();
      }
   // master -> slave communication
   void send(std::shared_ptr<socket> s, const void *buf, const size_t len);
   void send(std::shared_ptr<socket> s, const int x)
      {
      send(s, &x, sizeof(x));
      }
   void send(std::shared_ptr<socket> s, const double x)
      {
      send(s, &x, sizeof(x));
      }
   void send(std::shared_ptr<socket> s, const std::string& x)
      {
      int len = int(x.length());
      send(s, len);
      send(s, x.c_str(), len);
      }
   void call(std::shared_ptr<socket> s, const std::string& x)
      {
      send(s, int(tag_work));
      send(s, x);
      }
   void updatecputime(std::shared_ptr<socket> s);
   void receive(std::shared_ptr<socket> s, void *buf, const size_t len);
   void receive(std::shared_ptr<socket> s, int& x)
      {
      receive(s, &x, sizeof(x));
      }
   void receive(std::shared_ptr<socket> s, libbase::int64u& x)
      {
      receive(s, &x, sizeof(x));
      }
   void receive(std::shared_ptr<socket> s, double& x)
      {
      receive(s, &x, sizeof(x));
      }
   void receive(std::shared_ptr<socket> s, vector<double>& x);
   void receive(std::shared_ptr<socket> s, std::string& x);
};

} // end namespace

#endif
