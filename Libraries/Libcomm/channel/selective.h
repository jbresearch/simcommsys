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

/*!
 * \brief   Selective Channel
 * \author  Noel Farrugia
 *
 * Implements a selective channel based on the supplied bitmask. If the
 * bitmask's bit value is set to '1', the symbol will be transmitted over the
 * primary channel, otherwise it will be submitted via the secondary channel.
 * The secondary channel has a fixed error rate supplied by the user.
 *
 * The tests validating the implementation are found in the
 * \c Test/TestSelectiveChannel/testselectivechannel.cpp file.
 *
 * \tparam S Channel symbol type
 */

template <class S>
class selective : public channel<S, libbase::vector>
{
public:
    selective() = default;
    selective(const std::string& bitmask);
    selective(const std::string& bitmask,
              std::shared_ptr<channel<S>> primary_channel,
              std::shared_ptr<channel<S>> secondary_channel,
              const double secondary_channel_parameter);

    //! Sets the parameter of the primary channel.
    void set_parameter(const double x);
    //! Retrieves the parameter of the primary channel.
    double get_parameter() const;

    std::string get_bitmask() const;

    uint32_t get_num_tx_bits_on_primary_channel() const;
    uint32_t get_num_tx_bits_on_secondary_channel() const;

    const channel<S>& get_primary_channel() const;
    const channel<S>& get_secondary_channel() const;

    void seedfrom(libbase::random& r) override;

    /*!
     * \brief Simulates channel transmission.
     *
     * The transmission sequence \p tx is split based on the bitmask and
     * transmitted on a channel that is also specified by the bitmask.
     *
     * \param[in]  tx The sequence to transmit.
     * \param[out] rx The transmitted sequence.
     */
    void transmit(const libbase::vector<S>& tx,
                  libbase::vector<S>& rx) override;

    /*!
     * \brief Simulates channel reception.
     *
     * Calculates the probability of each received symbol out of all the valid
     * possible transmission symbols. These probabilities are calculated by the
     * primary and secondary channel implementations.
     *
     * \param[in]  possible_tx_symbols List of all possible transmitted symbols.
     * \param[in]  rx                  Received sequence.
     * \param[out] ptable              The probability table.
     */
    void
    receive(const libbase::vector<S>& possible_tx_symbols,
            const libbase::vector<S>& rx,
            libbase::vector<libbase::vector<double>>& ptable) const override;

    //! \copydoc channel::description
    std::string description() const;

private:
    //! Sets the bitmask and secondary channel's parameter.
    void init(const std::string& bitmask,
              const double secondary_channel_parameter);

    /*!
     * \brief Set the bitmask value.
     *
     * The \p bitmask string is validated and the number of bits to transmit on
     * each channel is calculated and stored. This is used in the
     * \ref split_sequence function for vector size reservation.
     *
     * \param bitmask The bitmask string.
     */
    void set_bitmask(const std::string& bitmask);

    //! This function is not implemented and should never be called.
    S corrupt(const S& s);
    //! This function is not implemented and should never be called.
    double pdf(const S& tx, const S& rx) const;

    /*!
     * \brief Splits a sequence based on bitmask.
     *
     * The primary sequence contains symbols that need to be transmitted over
     * the primary channel, while the secondary sequence contains the sequence
     * to transmit over the secondary channel.
     *
     * \param bitmask                          The bitmask.
     * \param num_tx_bits_on_primary_channel   The number of bits to transmit on
     *                                         the primary channel.
     * \param num_tx_bits_on_secondary_channel The number of bits to transmit on
     *                                         the secondary channel.
     * \param bit_sequence                     The bit sequence.
     *
     * \return Primary sequence, Secondary sequence pair.
     */
    static std::pair<libbase::vector<S>, libbase::vector<S>>
    split_sequence(const std::vector<bool>& bitmask,
                   const uint32_t num_tx_bits_on_primary_channel,
                   const uint32_t num_tx_bits_on_secondary_channel,
                   const libbase::vector<S>& bit_sequence);

    /*!
     * \brief Merge two sequences together based on bitmask.
     *
     * \param[in]  bitmask   The bitmask.
     * \param[in]  primary   The primary sequence.
     * \param[in]  secondary The secondary sequence.
     * \param[out] merged    The merged sequence.
     */
    static void merge_sequences(const std::vector<bool>& bitmask,
                                const libbase::vector<S>& primary,
                                const libbase::vector<S>& secondary,
                                libbase::vector<S>& merged);

    /*!
     * \brief Merge two probability tables (ptables) together based on bitmask.
     *
     * \param[in]  bitmask          The bitmask.
     * \param[in]  primary_ptable   The primary ptable.
     * \param[in]  secondary_ptable The secondary ptable.
     * \param[out] ptable           The merged ptable.
     */
    static void merge_ptables(
        const std::vector<bool>& bitmask,
        const libbase::vector<libbase::vector<double>>& primary_ptable,
        const libbase::vector<libbase::vector<double>>& secondary_ptable,
        libbase::vector<libbase::vector<double>>& ptable);

    /*!
     * \brief Ensures the bitmask contains only '1' or '0' characters.
     *
     * \param bitmask The bitmask.
     *
     * \throws libbase::load_error If bitmask is invalid.
     */
    static void validate_bitmask(const std::string& bitmask);

    /*!
     * \brief Ensures that the sequence and bitmask are of equal size.
     *
     * \param bitmask  The bitmask.
     * \param sequence The sequence.
     *
     * \throws std::runtime_error If sizes do not match.
     */
    static void validate_sequence_size(const std::vector<bool>& bitmask,
                                       const libbase::vector<S>& sequence);

    std::vector<bool> m_bitmask;
    uint32_t m_num_tx_bits_on_primary_channel;
    uint32_t m_num_tx_bits_on_secondary_channel;
    std::shared_ptr<channel<S>> m_primary_channel;
    std::shared_ptr<channel<S>> m_secondary_channel;

    // Serialization Support
    DECLARE_SERIALIZER(selective);
};

} // namespace libcomm

#endif /* SELECTIVE_CHANNEL_H */
