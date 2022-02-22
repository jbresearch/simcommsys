/*!
 * \file
 *
 * Copyright (c) 2022 Noel Farrugia
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

#ifndef SELECTIVE_CHANNEL_H
#define SELECTIVE_CHANNEL_H

#include "channel.h"

namespace libcomm
{

template <class Symbol>
class selective : public channel<Symbol>
{
public:
    selective() = default;

    selective(const std::string& bitstring);

    selective(const std::string& bitstring,
              std::shared_ptr<channel<Symbol>> channel);

    std::string get_bitmask() const;

    void transmit(const libbase::vector<Symbol>& tx,
                  libbase::vector<Symbol>& rx) override;

private:
    std::vector<bool>
    create_bitmask_from_bistring(const std::string& bitstring);

    Symbol corrupt(const Symbol& s);

    double pdf(const Symbol& tx, const Symbol& rx) const;

    void set_parameter(const double x);

    double get_parameter() const;

    std::string description() const;

    void validate_bitstring(const std::string& bitstring);

    std::vector<bool> m_bitmask;
    std::shared_ptr<channel<Symbol>> m_channel;

    // Serialization Support
    DECLARE_SERIALIZER(selective);
};

} // namespace libcomm

#endif /* SELECTIVE_CHANNEL_H */
