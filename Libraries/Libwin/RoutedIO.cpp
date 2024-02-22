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

#include "RoutedIO.h"
#include "stdafx.h"
#include <iostream>

namespace libwin
{

CRoutedIO::CRoutedIO()
{
    // route standard streams to trace/message box output
    std::ostream tracer(&m_tracer);
    std::ostream msgbox(&m_msgbox);
    std::clog.rdbuf(tracer.rdbuf());
    std::cout.rdbuf(tracer.rdbuf());
    std::cerr.rdbuf(msgbox.rdbuf());
}

} // namespace libwin
