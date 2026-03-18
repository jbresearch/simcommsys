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

#include "socket.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <netdb.h>
#include <unistd.h>

#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <arpa/inet.h>
#include <netinet/ip.h>

namespace libbase
{

// constant values

const int socket::connect_tries = 4;
const int socket::connect_delay = 10;

// static values

// helper functions

template <class T>
ssize_t
socket::io(T buf, size_t len)
{
    std::cerr << "Cannot instantiate template function with this type"
              << std::endl;
    exit(1);
    return 0;
}

template <>
ssize_t
socket::io(const void* buf, size_t len)
{
    return ::write(sd, buf, len);
}

template <>
ssize_t
socket::io(void* buf, size_t len)
{
    return ::read(sd, buf, len);
}

template <class T>
ssize_t
socket::insistio(T buf, size_t len)
{
    const char* b = (const char*)buf;
    size_t rem = len;

    do {
        ssize_t n = io(T(b), rem);

        if (n < 0) {
            return n;
        }

        if (n == 0) {
            return len - rem;
        }

        rem -= n;
        b += n;
    } while (rem);

    return len;
}

// constructor/destructor

socket::socket()
{
    sd = -1;
    listener = true;
}

socket::~socket()
{
    if (sd >= 0) {
        trace << "DEBUG (~socket): closing socket " << sd << std::endl;
        close(sd);
    }
}

// wait for client connects

bool
socket::bind(uint16_t port)
{
    if ((sd = (int)::socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0) {
        std::cerr << "ERROR (bind): Failed to create socket descriptor"
                  << std::endl;
        return false;
    }

    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = htonl(INADDR_ANY);
    sin.sin_port = htons(port);

    int opt = 1;
    if (setsockopt(sd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        std::cerr << "ERROR (bind): Failed to set socket options" << std::endl;
        return false;
    }

    if (::bind(sd, (struct sockaddr*)&sin, sizeof(struct sockaddr_in))) {
        std::cerr << "ERROR (bind): Failed to bind socket options" << std::endl;
        return false;
    }

    if (listen(sd, 5)) {
        std::cerr << "ERROR (bind): Failure on listening for connections"
                  << std::endl;
        return false;
    }

    // The socket is now ready to accept() connections
    trace << "DEBUG (bind): Bound to socket, ready to accept connections"
          << std::endl;

    return true;
}

std::list<std::shared_ptr<socket>>
socket::select(std::list<std::shared_ptr<socket>> sl, const double timeout)
{
    fd_set rfds;
    FD_ZERO(&rfds);

    int max = 0;
    for (std::list<std::shared_ptr<socket>>::iterator i = sl.begin();
         i != sl.end();
         ++i) {
        FD_SET((*i)->sd, &rfds);
        if ((*i)->sd > max) {
            max = (*i)->sd;
        }
    }
    ++max;

    struct timeval s_timeout;
    s_timeout.tv_sec = int(floor(timeout));
    s_timeout.tv_usec = int((timeout - floor(timeout)) * 1E6);
    ::select(max, &rfds, NULL, NULL, timeout == 0 ? NULL : &s_timeout);

    std::list<std::shared_ptr<socket>> al;
    for (std::list<std::shared_ptr<socket>>::iterator i = sl.begin();
         i != sl.end();
         ++i) {
        if (FD_ISSET((*i)->sd, &rfds)) {
            al.push_back(*i);
        }
    }

    return al;
}

std::shared_ptr<socket>
socket::accept()
{
    std::shared_ptr<socket> s(new socket);
    socklen_t len = sizeof(struct sockaddr_in);
    struct sockaddr_in clnt;
    s->sd = (int)::accept(sd, (struct sockaddr*)&clnt, &len);

    if (s->sd < 0) {
        std::cerr << "ERROR (accept): Failure on listening for connections"
                  << std::endl;
        exit(1);
    }

    s->ip = inet_ntoa(clnt.sin_addr);
    s->port = ntohs(clnt.sin_port);
    s->listener = false;
    trace << "DEBUG (accept): Accepted new client from " << s->ip << ":"
          << s->port << std::endl;

    return s;
}

// open connection to server, returning the local port

uint16_t
socket::connect(std::string hostname, uint16_t port)
{
    if ((sd = (int)::socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0) {
        std::cerr << "ERROR (connect): Failed to create socket descriptor"
                  << std::endl;
        return 0;
    }

    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);

    // Do a DNS lookup of the hostname and try to connect
    struct hostent* hp;
    if (!(hp = gethostbyname(hostname.c_str()))) {
        std::cerr << "ERROR (connect): Failed to resolve host address"
                  << std::endl;
        return 0;
    }

    memcpy(&sin.sin_addr, hp->h_addr_list[0], sizeof(struct in_addr));

    for (int i = 1; i <= connect_tries; i++) {
        if (::connect(sd, (struct sockaddr*)&sin, sizeof(struct sockaddr_in)) ==
            0) {
            break;
        }

        std::cerr << "WARNING (connect): Connect failed, try " << i << " of "
                  << connect_tries << std::endl;

        if (i == connect_tries) {
            std::cerr << "ERROR (connect): Too many connection failures"
                      << std::endl;
            return 0;
        } else {
            sleep(connect_delay);
        }
    }

    // determine the local port for this connection
    struct sockaddr_in local_sin;
    socklen_t addr_len = sizeof(local_sin);
    if (getsockname(sd, (struct sockaddr*)&local_sin, &addr_len) != 0) {
        std::cerr << "ERROR: Could not retrieve local port info" << std::endl;
        return 0;
    }
    uint16_t local_port = ntohs(local_sin.sin_port);

    // TCP/IP connection has been established
    trace << "DEBUG (connect): Connection to " << hostname << ":" << port
          << " established from port " << local_port << std::endl;
    socket::ip = hostname;
    socket::port = port;
    return local_port;
}

// read/write data
// NOTE: these cannot be moved to the header or they will use the templated
// methods before the specialized versions are defined.

ssize_t
socket::write(const void* buf, size_t len)
{
    return io(buf, len);
}

ssize_t
socket::read(void* buf, size_t len)
{
    return io(buf, len);
}

bool
socket::insistwrite(const void* buf, size_t len)
{
    return insistio(buf, len) == ssize_t(len);
}

bool
socket::insistread(void* buf, size_t len)
{
    return insistio(buf, len) == ssize_t(len);
}

} // namespace libbase
