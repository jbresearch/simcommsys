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

#include "masterslave.h"

#include "pacifier.h"
#include "timer.h"
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#    include <winsock2.h>
#else
#    include <signal.h>
#    include <sys/resource.h>
#    include <sys/time.h>
#    include <unistd.h>
#endif

#ifndef HOST_NAME_MAX
#    define HOST_NAME_MAX 255
#endif

namespace libbase
{

// items for use by everyone

void
masterslave::fregister(const std::string& name, std::shared_ptr<functor> f)
{
    trace << "DEBUG: Register function \"" << name << "\" - ";
    fmap[name] = f;
    trace << fmap.size() << " functions registered, done." << std::endl;
}

void
masterslave::fcall(const std::string& name)
{
    trace << "DEBUG: Call function \"" << name << "\" - ";
    typename std::map<std::string, std::shared_ptr<functor>>::iterator search =
        fmap.find(name);
    if (search == fmap.end()) {
        std::cerr << "Function \"" << name << "\" unknown, cannot continue."
                  << std::endl;
        exit(1);
    }
    try {
        search->second->call();
    } catch (std::runtime_error& e) {
        std::cerr << "Runtime exception: " << e.what() << std::endl;
        exit(1);
    }
    trace << "done." << std::endl;
}

/*! \brief Global enable/disable of master-slave system
 *
 * Returns the operating mode used; for local and master, return is immediate.
 * For slaves, return is only when the slave dies gracefully.
 *
 * \note Endpoint can be:
 * - 'local', indicating local-computation model; in this case, the
 * class will not be initialized.
 * - ':port', indicating server-mode, bound to given port
 * - 'hostname:port', indicating client-mode, connecting to given
 * host/port combination
 */
masterslave::mode_t
masterslave::enable(const std::string& endpoint, bool quiet, int priority)
{
    assert(!initialized);

#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif

    // hostname is the part before the ':', or the whole string if there is no
    // ':'
    const size_t n = endpoint.find(':');
    const std::string hostname = endpoint.substr(0, n);

    // port is the part after the ':', if there was one
    // if there is no port, hostname must be 'local'
    int port = 0;
    if (n != std::string::npos) {
        std::istringstream(endpoint.substr(n + 1)) >> port;
    }

    // interpret quiet flag
    if (quiet) {
        pacifier::disable_output();
    }

    // Handle option for local computation only
    if (hostname.compare("local") == 0 && port == 0) {
        trace << "Using local computation" << std::endl;
        // start timers
        twall.start();
        tcpu.start();
        return mode_local;
    } else if (hostname.length() > 0) {
        // If the hostname part isn't empty, it's a slave process
        slaveprocess(hostname, port, priority);
        return mode_slave;
    } else {
        // Otherwise, this must be the master process.
        master.reset(new socket);
        assertalways(master->bind(port));
        trace << "Master system bound to port " << port << std::endl;
        initialized = true;
        // start timers
        twall.start();
        tcpu.start();
        return mode_master;
    }

    // we should never get here
}

// static items (for use by slaves)

void
masterslave::close()
{
    std::cerr << "Losing connection with master [" << master->getip() << ":"
              << master->getport() << "]" << std::endl;
    master.reset();
}

void
masterslave::setpriority(const int priority)
{
#ifdef _WIN32
#else
    const int PRIO_CURRENT = 0;
    ::setpriority(PRIO_PROCESS, PRIO_CURRENT, priority);
#endif
}

void
masterslave::connect(const std::string& hostname, const int16u port)
{
    std::cerr << "Connecting to " << hostname << ":" << port << std::endl;
    master.reset(new socket);
    if (!master->connect(hostname, port)) {
        std::cerr << "Connection failed, giving up." << std::endl;
        exit(1);
    }
}

std::string
masterslave::gethostname()
{
    const int len = HOST_NAME_MAX + 1;
    char hostname[len];
    ::gethostname(hostname, len);
    return hostname;
}

int
masterslave::gettag()
{
    walltimer tslave("masterslave_slave");
    int tag;
    receive(tag);
    tslave.stop();
    trace << "Slave latency = " << tslave << ": ";
    return tag;
}

void
masterslave::sendname()
{
    std::string hostname = gethostname();
    send(hostname);
    trace << "send hostname [" << hostname << "]" << std::endl;
}

void
masterslave::sendcputime()
{
    const double cputime = tcpu.elapsed();
    tcpu.start();
    send(cputime);
    cputimeused += cputime;
    trace << "send usage [" << cputime << "]" << std::endl;
}

void
masterslave::dowork()
{
    std::string key;
    receive(key);
    trace << "system working" << std::endl;
    fcall(key);
}

void
masterslave::slaveprocess(const std::string& hostname,
                          const int16u port,
                          const int priority)
{
    setpriority(priority);
    connect(hostname, port);
    // Status information for user
    std::cerr << "Slave system starting at priority " << priority << "."
              << std::endl;
    // infinite loop, until we are explicitly told to die
    twall.start();
    tcpu.start();
    try {
        while (true) {
            const int tag = gettag();
            switch (tag) {
            case tag_getname:
                sendname();
                break;
            case tag_getcputime:
                sendcputime();
                break;
            case tag_work:
                dowork();
                break;
            case tag_die:
                twall.stop();
                tcpu.stop();
                std::cerr << "Received die request, stopping after "
                          << timer::format(getcputime()) << " CPU runtime ("
                          << int(100 * getusage()) << "% usage)." << std::endl;
                close();
                return;
            default:
                std::cerr << "received bad tag [" << tag << "]" << std::endl;
                exit(1);
            }
        }
    } catch (std::runtime_error& e) {
        std::cerr << "Runtime exception: " << e.what() << std::endl;
        exit(1);
    }
}

// slave -> master communication

void
masterslave::send(const void* buf, const size_t len)
{
    if (!master->insistwrite(buf, len)) {
        std::ostringstream sstr;
        sstr << "Write failed to [" << master->getip() << ":"
             << master->getport() << "]";
        close();
        throw std::runtime_error(sstr.str());
    }
}

/*! \brief Send a vector<double> to the master
 * \note Vector size is sent first; this makes foreknowledge of size and
 * pre-initialization unnecessary.
 */
void
masterslave::send(const vector<double>& x)
{
    const int count = x.size();
    send(count);
    send(&x(0), sizeof(double) * count);
}

void
masterslave::send(const std::string& x)
{
    int len = int(x.length());
    send(len);
    send(x.c_str(), len);
}

void
masterslave::receive(void* buf, const size_t len)
{
    if (!master->insistread(buf, len)) {
        std::ostringstream sstr;
        sstr << "Read failed from [" << master->getip() << ":"
             << master->getport() << "]";
        close();
        throw std::runtime_error(sstr.str());
    }
}

void
masterslave::receive(std::string& x)
{
    int len;
    receive(len);
    std::vector<char> buf(len);
    receive(&buf[0], len);
    x.assign(&buf[0], len);
}

// non-static items (for use by master)

void
masterslave::close(std::shared_ptr<socket> s)
{
    std::cerr << "Slave [" << s->getip() << ":" << s->getport() << "] gone";
    smap.erase(s);
    std::cerr << ", currently have " << smap.size() << " clients" << std::endl;
}

// disable function

/*! \brief Shuts down master-slave system
 *
 * \todo Specify what happens if the system was never initialized
 */
void
masterslave::disable()
{
    // if the master-slave system is not initialized, there is nothing to do
    if (!initialized) {
        return;
    }

    // kill all remaining slaves
    std::clog << "Killing idle slaves:" << std::flush;
    while (std::shared_ptr<socket> s = find_idle_slave()) {
        trace << "DEBUG (disable): Idle slave found (" << s << "), killing."
              << std::endl;
        std::clog << "." << std::flush;
        send(s, int(tag_die));
    }
    // TODO: wait for slaves to end gracefully or timeout
    std::clog << " done" << std::endl;
    // print timer information
    twall.stop();
    tcpu.stop();
    std::clog << "Time elapsed: " << twall << "" << std::endl;
    std::clog << "CPU usage on master: "
              << int(100 * tcpu.elapsed() / twall.elapsed()) << "%"
              << std::endl;
    std::clog.precision(2);
    std::clog << "Average speedup factor: " << getusage() << "" << std::endl;
    // update flag
    initialized = false;
}

// slave-interface functions

std::shared_ptr<socket>
masterslave::find_new_slave()
{
    for (std::map<std::shared_ptr<socket>, state_t>::iterator i = smap.begin();
         i != smap.end();
         ++i)
        if (i->second == state_new) {
            i->second = state_idle;
            return i->first;
        }
    return std::shared_ptr<socket>();
}

std::shared_ptr<socket>
masterslave::find_idle_slave()
{
    for (std::map<std::shared_ptr<socket>, state_t>::iterator i = smap.begin();
         i != smap.end();
         ++i)
        if (i->second == state_idle) {
            i->second = state_working;
            return i->first;
        }
    return std::shared_ptr<socket>();
}

std::shared_ptr<socket>
masterslave::find_pending_slave()
{
    for (std::map<std::shared_ptr<socket>, state_t>::iterator i = smap.begin();
         i != smap.end();
         ++i)
        if (i->second == state_eventpending) {
            i->second = state_idle;
            return i->first;
        }
    return std::shared_ptr<socket>();
}

/*! \brief Number of slaves currently in 'working' state
 */
int
masterslave::count_workingslaves() const
{
    int count = 0;
    for (std::map<std::shared_ptr<socket>, state_t>::const_iterator i =
             smap.begin();
         i != smap.end();
         ++i)
        if (i->second == state_working) {
            count++;
        }
    return count;
}

bool
masterslave::anyoneworking() const
{
    for (std::map<std::shared_ptr<socket>, state_t>::const_iterator i =
             smap.begin();
         i != smap.end();
         ++i)
        if (i->second == state_working) {
            return true;
        }
    return false;
}

/*! \brief Waits for a socket event
 * \param acceptnew Flag to indicate whether new connections are allowed
 * (defaults to true)
 * \param timeout Return with no event if this many seconds elapses (zero
 * means wait forever; this is the default)
 */
void
masterslave::waitforevent(const bool acceptnew, const double timeout)
{
    static bool firsttime = true;
    if (firsttime) {
        std::cerr << "Master system ready; waiting for clients." << std::endl;
        firsttime = false;
    }

    static bool signalentry = true;
    if (signalentry) {
        trace << "DEBUG (estimate): Waiting for event." << std::endl;
        signalentry = false;
    }

    // create list of sockets and select
    std::list<std::shared_ptr<socket>> sl, al;

    sl.push_back(master);
    for (std::map<std::shared_ptr<socket>, state_t>::iterator i = smap.begin();
         i != smap.end();
         ++i) {
        sl.push_back(i->first);
    }

    al = socket::select(sl, timeout);
    if (!al.empty()) {
        signalentry = true;
    }

    for (std::list<std::shared_ptr<socket>>::iterator i = al.begin();
         i != al.end();
         ++i) {
        if ((*i)->islistener() && acceptnew) {
            std::shared_ptr<socket> newslave = (*i)->accept();
            smap[newslave] = state_new;
            std::cerr << "New slave [" << newslave->getip() << ":"
                      << newslave->getport() << "], currently have "
                      << smap.size() << " clients" << std::endl;
        } else {
            smap[*i] = state_eventpending;
        }
    }
}

/*!
 * \brief Reset given slave to the 'new' state
 *
 * \note Slave must be in the 'idle' state
 */
void
masterslave::resetslave(std::shared_ptr<socket> s)
{
    assertalways(smap[s] == state_idle);
    smap[s] = state_new;
}

/*!
 * \brief Reset all 'idle' slaves to the 'new' state
 */
void
masterslave::resetslaves()
{
    while (std::shared_ptr<socket> s = find_idle_slave()) {
        smap[s] = state_new;
    }
}

// master -> slave communication

void
masterslave::send(std::shared_ptr<socket> s, const void* buf, const size_t len)
{
    if (!s->insistwrite(buf, len)) {
        std::ostringstream sstr;
        sstr << "Write failed to [" << s->getip() << ":" << s->getport() << "]";
        close(s);
        throw std::runtime_error(sstr.str());
    }
}

/*! \brief Accumulate CPU time for given slave
 * \param s Slave from which to get CPU time
 */
void
masterslave::updatecputime(std::shared_ptr<socket> s)
{
    double cputime;
    send(s, int(tag_getcputime));
    receive(s, cputime);
    cputimeused += cputime;
}

void
masterslave::receive(std::shared_ptr<socket> s, void* buf, const size_t len)
{
    if (!s->insistread(buf, len)) {
        std::ostringstream sstr;
        sstr << "Read failed from [" << s->getip() << ":" << s->getport()
             << "]";
        close(s);
        throw std::runtime_error(sstr.str());
    }
}

/*! \brief Receive a vector<double> from given slave
 * \note Vector size is obtained first; this makes foreknowledge of size and
 * pre-initialization unnecessary.
 */
void
masterslave::receive(std::shared_ptr<socket> s, vector<double>& x)
{
    // get vector size first
    int count;
    receive(s, count);
    // initialize vector and get vector elements
    x.init(count);
    receive(s, &x(0), sizeof(double) * count);
}

void
masterslave::receive(std::shared_ptr<socket> s, std::string& x)
{
    int len;
    receive(s, len);
    std::vector<char> buf(len);
    receive(s, &buf[0], len);
    x.assign(&buf[0], len);
}

} // namespace libbase
// end namespace
