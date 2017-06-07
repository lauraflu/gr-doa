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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "MUSIC_lin_array_cnx_impl.h"

#define COPY_MEM false  // Do not copy matrices into separate memory
#define FIX_SIZE true   // Keep dimensions of matrices constant

namespace gr {
  namespace doa {

    MUSIC_lin_array_cnx::sptr
    MUSIC_lin_array_cnx::make(
      float norm_spacing,
      int num_targets,
      int num_ant_ele,
      int pspectrum_len,
      std::string distributionFIFO,
      std::string reductionFIFO,
      std::string writeFIFO,
      std::string readFIFO)
    {
      return gnuradio::get_initial_sptr
        (new MUSIC_lin_array_cnx_impl(norm_spacing, num_targets, num_ant_ele,
          pspectrum_len, distributionFIFO, reductionFIFO, writeFIFO, readFIFO));
    }

    /*
     * The private constructor
     */
    MUSIC_lin_array_cnx_impl::MUSIC_lin_array_cnx_impl(
      float norm_spacing,
      int num_targets,
      int num_ant_ele,
      int pspectrum_len,
      std::string distributionFIFO,
      std::string reductionFIFO,
      std::string writeFIFO,
      std::string readFIFO)
      : gr::sync_block("MUSIC_lin_array_cnx",
              gr::io_signature::make(1, 1, sizeof(gr_complex) * num_ant_ele * num_ant_ele),
              gr::io_signature::make(1, 1, sizeof(float) * pspectrum_len)),
              d_norm_spacing(norm_spacing),
              d_num_targets(num_targets),
              d_num_ant_ele(num_ant_ele),
              d_pspectrum_len(pspectrum_len)
      , arr_size(num_ant_ele)
      , nr_arrays(pspectrum_len)
    {
        // Variables calculated based on the input parameters
        mat_size = arr_size * arr_size;
        arr_size_c = arr_size * 2;
        mat_size_c = mat_size * 2;
        nr_arrays_elems = nr_arrays * arr_size;
        arr_in_chunk = (process_at_once * vector_array_size) / mat_size_c;
        nr_chunks = nr_arrays / arr_in_chunk;
        nr_elem_chunk = arr_in_chunk * arr_size;
        nr_elem_calc = process_at_once * vector_array_size / arr_size_c;

        // Create ConnexMachine instance
        try {
          connex = new ConnexMachine(distributionFIFO,
                                     reductionFIFO,
                                     writeFIFO,
                                     readFIFO);
        } catch (std::string err) {
          std::cout << err << std::endl;
        }

        factor_mult1 = 1 << 14;
        factor_mult2 = 1 << 16;
        factor_res = 1 << 14;

        const int blocks_to_reduce = vector_array_size / arr_size_c;
        const int size_of_block = arr_size_c;

        // Create the kernel
        multiply_kernel(process_at_once, size_of_block, blocks_to_reduce);

        // form antenna array locations centered around zero and normalize
        d_array_loc = fcolvec(d_num_ant_ele, fill::zeros);
        for (int nn = 0; nn < d_num_ant_ele; nn++)
        {
            d_array_loc(nn) = d_norm_spacing*0.5*(d_num_ant_ele-1-2*nn);
        }

        // form theta vector
        d_theta = new float[d_pspectrum_len];
        d_theta[0] = 0.0;
        float theta_prev = 0.0, theta;
        for (int ii = 1; ii < d_pspectrum_len; ii++)
        {
          theta = theta_prev+180.0/d_pspectrum_len;
          theta_prev = theta;
          d_theta[ii] = datum::pi*theta/180.0;
        }

        // form array response matrix
        cx_fcolvec vii_temp(d_num_ant_ele, fill::zeros);
        d_vii_matrix = cx_fmat(d_num_ant_ele,d_pspectrum_len);
        d_vii_matrix_trans = cx_fmat(d_pspectrum_len,d_num_ant_ele);
        for (int ii = 0; ii < d_pspectrum_len; ii++)
        {
          // generate array manifold vector for each theta
          amv(vii_temp, d_array_loc, d_theta[ii]);
          // add as column to matrix
          d_vii_matrix.col(ii) = vii_temp;
        }
        // save transposed copy
        d_vii_matrix_trans = trans(d_vii_matrix);
    }

    /*
     * Our virtual destructor.
     */
    MUSIC_lin_array_cnx_impl::~MUSIC_lin_array_cnx_impl()
    {
      delete connex;
    }

    // array manifold vector generating function
    void MUSIC_lin_array_cnx_impl::amv(cx_fcolvec& v_ii, fcolvec& array_loc, float theta)
    {
        // sqrt(-1)
        const gr_complex i = gr_complex(0.0, 1.0);
        // array manifold vector
        v_ii = exp(i * (-1.0 * 2 * datum::pi * cos(theta) * array_loc));
    }


    int
    MUSIC_lin_array_cnx_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *)input_items[0];
      float *out = (float *) output_items[0];

      // process each input vector (Rxx matrix)
      fvec eig_val;
      cx_fmat eig_vec;
      cx_fmat U_N;
      cx_fmat U_N_sq;
      for (int item = 0; item < noutput_items; item++)
      {
          // make input pointer into matrix pointer
          cx_fmat in_matrix(in+item*d_num_ant_ele*d_num_ant_ele, d_num_ant_ele, d_num_ant_ele);
          fvec out_vec(out+item*d_pspectrum_len, d_pspectrum_len, COPY_MEM, FIX_SIZE);

          // determine EVD of the auto-correlation matrix
          eig_sym(eig_val, eig_vec, in_matrix);

          // noise subspace and its square matrix
          U_N = eig_vec.cols(0, d_num_ant_ele-d_num_targets-1);

          U_N_sq = U_N*trans(U_N);

          // determine pseudo-spectrum for each value of theta in [0.0, 180.0)
          gr_complex Q_temp;
          for (int ii = 0; ii < d_pspectrum_len; ii++)
          {
            Q_temp = as_scalar(d_vii_matrix_trans.row(ii)*U_N_sq*d_vii_matrix.col(ii));
            out_vec(ii) = 1.0/Q_temp.real();
          }
          out_vec = 10.0*log10(out_vec/out_vec.max());
      }

      // Tell runtime system how many output items we produced.
      return noutput_items;

    }

    /*===================================================================
     * Define ConnexArray kernels that will be used in the worker
     *===================================================================*/
    int MUSIC_lin_array_cnx_impl::executeMultiplyArrMat(ConnexMachine *connex)
    {
      try {
        connex->executeKernel("multiply_arr_mat");
      } catch (std::string e) {
        std::cout << e << std::endl;
        return -1;
      }
      return 0;
    }

    void MUSIC_lin_array_cnx_impl::multiply_kernel(
      int process_at_once, int size_of_block, int blocks_to_reduce)
    {
      BEGIN_KERNEL("multiply_arr_mat");
        EXECUTE_IN_ALL(
          R25 = 0;
          R26 = 511;
          R30 = 1;
          R31 = 0;
          R28 = size_of_block;  // Equal to ARR_SIZE_C; dimension of the blocks
                                // on which reduction is performed at once
        )

        EXECUTE_IN_ALL(
          R1 = LS[R25];           // z1 = a1 + j * b1
          R2 = LS[R26];           // z2 = a2 + j * b2
          R29 = INDEX;          // Used later to select PEs for reduction
          R27 = size_of_block;  // Used to select blocks of ARR_SIZE_C for reduction

          R3 = R1 * R2;         // a1 * a2, b1 * b2
          R3 = MULT_HIGH();

          CELL_SHL(R2, R30);    // Bring b2 to the left to calc b2 * a1
          NOP;
          R4 = SHIFT_REG;
          R4 = R1 * R4;         // a1 * b2
          R4 = MULT_HIGH();

          CELL_SHR(R2, R30);
          NOP;
          R5 = SHIFT_REG;
          R5 = R1 * R5;         // b1 * a2
          R5 = MULT_HIGH();

          R9 = INDEX;           // Select only the odd PEs
          R9 = R9 & R30;
          R7 = (R9 == R30);
        )

        EXECUTE_WHERE_EQ(       // Only in the odd PEs
          // Getting -b1 * b2 in each odd cell
          R3 = R31 - R3;        // All partial real parts are in R3

          R4 = R5;              // All partial imaginary parts are now in R4
        )

        REPEAT_X_TIMES(blocks_to_reduce);
          EXECUTE_IN_ALL(
            R7 = (R29 < R27);   // Select only blocks of 8 PEs at a time by
                                // checking that the index is < k * 8
          )
          EXECUTE_WHERE_LT(
            R29 = 129;          // A random number > 128 so these PEs won't be
                                // selected again
            REDUCE(R3);         // Real part
            REDUCE(R4);         // Imaginary part
          )
          EXECUTE_IN_ALL(
            R27 = R27 + R28;    // Go to the next block of 8 PEs
          )
        END_REPEAT;

      END_KERNEL("multiply_arr_mat");
    }


  } /* namespace doa */
} /* namespace gr */

