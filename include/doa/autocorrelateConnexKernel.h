/* -*- c++ -*- */
/* 
 * Copyright 2017 <+YOU OR YOUR COMPANY+>.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */


#ifndef INCLUDED_DOA_AUTOCORRELATECONNEXKERNEL_H
#define INCLUDED_DOA_AUTOCORRELATECONNEXKERNEL_H

#include <doa/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace doa {

    /*!
     * \brief <+description of block+>
     * \ingroup doa
     *
     */
    class DOA_API autocorrelateConnexKernel : virtual public gr::block
    {
     public:
      typedef boost::shared_ptr<autocorrelateConnexKernel> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of doa::autocorrelateConnexKernel.
       *
       * To avoid accidental use of raw pointers, doa::autocorrelateConnexKernel's
       * constructor is in a private implementation
       * class. doa::autocorrelateConnexKernel::make is the public interface for
       * creating new instances.
       */
      static sptr make(int inputs, int snapshot_size, int overlap_size, int avg_method, std::string distributionFIFO, std::string reductionFIFO, std::string writeFIFO, std::string readFIFO);
    };

  } // namespace doa
} // namespace gr

#endif /* INCLUDED_DOA_AUTOCORRELATECONNEXKERNEL_H */

