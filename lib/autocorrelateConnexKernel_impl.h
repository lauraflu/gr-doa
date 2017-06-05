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

#ifndef INCLUDED_DOA_AUTOCORRELATECONNEXKERNEL_IMPL_H
#define INCLUDED_DOA_AUTOCORRELATECONNEXKERNEL_IMPL_H

#include <doa/autocorrelateConnexKernel.h>
#include <armadillo>
#include "ConnexMachine.h"

namespace gr {
  namespace doa {

    class autocorrelateConnexKernel_impl : public autocorrelateConnexKernel
    {
     private:
      const int d_num_inputs;
      const int d_snapshot_size;
      const int d_overlap_size;
      const int d_avg_method; // value assigned using the initialization list of the constructor
      int d_nonoverlap_size;
      arma::cx_fmat d_J;
      arma::cx_fmat d_input_matrix;

      /*====================================================================
       * CONNEX RELATED
       *==================================================================*/

      ConnexMachine *connex;

      /*
       * Factors required for scaling the data
       */
      int factor_mult;
      int factor_res;

      /*
       * Dimensions of the matrix for which the autocorrelation is calculated
       * ATTENTION! For a matrix Xk, the correlation is calculated as 1/k * Xk *
       * Xk^H, and n_rows and n_cols are its dimensions.
       * However, the input matrix of the block is considered to be stored
       * column-first and it is the transpose of Xk (in order to immitate the
       * situation in the gr-doa project).
       */
      int n_rows;
      int n_cols;

      /*
       * ConnexArray specific constants
       */
      const int vector_array_size = 128;
      const int local_storage_size = 1024;

      /*
       * There must be space for at least 2 * n_elems_in at the address pointed
       * by out_data.
       */
      void prepareInData(
        uint16_t *out_data,
        const gr_complex *in_data,
        const int n_elems_in);

      /*
       * There must be space for at least n_elems_in / 2 at the address pointed
       * by out_data.
       */
      void prepareOutData(
        gr_complex *out_data,
        const int32_t *in_data,
        const int n_elems_in
      );

      void printOutData(
        const uint16_t *in_data,
        const int n_elems_in
      );

      gr_complex prepareAndProcessOutData(
        const int32_t *in_data, const int n_elems_in);

     public:
      autocorrelateConnexKernel_impl(
        int inputs,
        int snapshot_size,
        int overlap_size,
        int avg_method,
        std::string distributionFIFO,
        std::string reductionFIFO,
        std::string writeFIFO,
        std::string readFIFO);
      ~autocorrelateConnexKernel_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
    };

  } // namespace doa
} // namespace gr

#endif /* INCLUDED_DOA_AUTOCORRELATECONNEXKERNEL_IMPL_H */

