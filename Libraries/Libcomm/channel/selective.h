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

template <class S>
class selective : public channel<S>
{
public:
    selective() = default;
    selective(const std::string& bitmask);
    selective(const std::string& bitmask,
              std::shared_ptr<channel<S>> primary_channel,
              std::shared_ptr<channel<S>> secondary_channel,
              const double secondary_channel_parameter);

    void set_parameter(const double x);
    double get_parameter() const;

    void set_bitmask(const std::string& bitmask);
    std::string get_bitmask() const;

    const channel<S>& get_primary_channel() const;

    const channel<S>& get_secondary_channel() const;

    void transmit(const libbase::vector<S>& tx,
                  libbase::vector<S>& rx) override;

    void
    receive(const libbase::vector<S>& possible_tx_symbols,
            const libbase::vector<S>& rx,
            libbase::vector<libbase::vector<double>>& ptable) const override;

    std::string description() const;

private:
    void init(const std::string& bitmask,
              const double secondary_channel_parameter);

    std::pair<libbase::vector<S>, libbase::vector<S>>
    split_sequence(const libbase::vector<S>& bit_sequence) const;

    void merge_sequences(const libbase::vector<S>& primary,
                         const libbase::vector<S>& secondary,
                         libbase::vector<S>& merged) const;

    void merge_ptables(
        const libbase::vector<libbase::vector<double>>& primary_ptable,
        const libbase::vector<libbase::vector<double>>& secondary_ptable,
        libbase::vector<libbase::vector<double>>& ptable) const;

    void validate_bitmask(const std::string& bitmask);
    void validate_sequence_size(const libbase::vector<S>& sequence) const;

    S corrupt(const S& s);
    double pdf(const S& tx, const S& rx) const;

    std::vector<bool> m_bitmask;
    std::shared_ptr<channel<S>> m_primary_channel;
    std::shared_ptr<channel<S>> m_secondary_channel;

    // Serialization Support
    DECLARE_SERIALIZER(selective);
};

} // namespace libcomm

#endif /* SELECTIVE_CHANNEL_H */
