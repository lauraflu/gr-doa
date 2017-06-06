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

#ifndef INCLUDED_DOA_MUSIC_LIN_ARRAY_CNX_IMPL_H
#define INCLUDED_DOA_MUSIC_LIN_ARRAY_CNX_IMPL_H

#include <doa/MUSIC_lin_array_cnx.h>

#include <armadillo>
using namespace arma;

namespace gr {
  namespace doa {

    class MUSIC_lin_array_cnx_impl : public MUSIC_lin_array_cnx
    {
     private:
      float d_norm_spacing;
      int d_num_targets;
      int d_num_ant_ele;
      int d_pspectrum_len;
      float *d_theta;
      fcolvec d_array_loc;
      cx_fmat d_vii_matrix;
      cx_fmat d_vii_matrix_trans;

     public:
      void amv(cx_fcolvec& v_ii, fcolvec& array_loc, float theta);

      MUSIC_lin_array_cnx_impl(
        float norm_spacing,
        int num_targets,
        int num_ant_ele,
        int pspectrum_len,
        std::string distributionFIFO,
        std::string reductionFIFO,
        std::string writeFIFO,
        std::string readFIFO);

      ~MUSIC_lin_array_cnx_impl();

      // Where all the action really happens
      int work(int noutput_items,
         gr_vector_const_void_star &input_items,
         gr_vector_void_star &output_items);
    };

  } // namespace doa
} // namespace gr

#endif /* INCLUDED_DOA_MUSIC_LIN_ARRAY_CNX_IMPL_H */

