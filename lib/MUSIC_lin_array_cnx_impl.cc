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

        if (arr_process_at_once > nr_arrays) {
          std::cout << "There are more arrays in a processing than arrays available!" << std::endl;
          std::cout << "Choose a number smaller than " << nr_arrays << std::endl;
          return;
        }

        int arrays_per_LS = vector_array_size / mat_size_c;
        process_at_once = arr_process_at_once / arrays_per_LS;
        arr_per_chunk = process_at_once * arrays_per_LS;
        nr_chunks = nr_arrays / arr_per_chunk;
        // By calculated element we mean one element from an output array that
        // is the result of an arr * mat multiplication
        nr_elem_calc = process_at_once * (vector_array_size / arr_size_c);
        nr_elem_calc_c = 2 * nr_elem_calc; // real and imaginary parts

        // Create ConnexMachine instance
        try {
          connex = new ConnexMachine(distributionFIFO,
                                     reductionFIFO,
                                     writeFIFO,
                                     readFIFO);
        } catch (std::string err) {
          std::cout << err << std::endl;
        }

        factor_mult1 = 1 << 15;
        factor_mult2 = 1 << 15;
        factor_res = 1 << 15;

        // Number of blocks to reduce in a single LS
        const int blocks_to_reduce = vector_array_size / arr_size_c;
        const int size_of_block = arr_size_c;

        // Create the kernel
        try {
          init_kernel(size_of_block);
          init_index();
          multiply_kernel(process_at_once, size_of_block, blocks_to_reduce);
        } catch (std::string e) {
          std::cout << e << std::endl;
        }

        executeLocalKernel(connex, "initKernel");

        // Allocate memory for the data that will be passed to the ConnexArray
        // and the data that it produces.
        in0_i = static_cast<uint16_t *>
            (malloc(nr_chunks * process_at_once * vector_array_size * sizeof(uint16_t)));
        in1_i = static_cast<uint16_t *>
            (malloc(vector_array_size * sizeof(uint16_t)));
        res_mult = static_cast<int32_t *>
            (malloc(nr_chunks * nr_elem_calc_c * sizeof(int32_t)));

        if ((in0_i == NULL) || (in1_i == NULL) || (res_mult == NULL)) {
          std::cout << "Malloc error at in0_i/in1_i/res_mult!" << std::endl;
        }

        // Form matrix that will store the temporary results in a chunk
        res_temp = cx_fmat(arr_per_chunk, arr_size, fill::zeros);

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
//        d_vii_matrix_trans = cx_fmat(d_pspectrum_len,d_num_ant_ele);
        for (int ii = 0; ii < d_pspectrum_len; ii++)
        {
          // generate array manifold vector for each theta
          amv(vii_temp, d_array_loc, d_theta[ii]);
          // add as column to matrix
          d_vii_matrix.col(ii) = vii_temp;
        }
        // save transposed copy
        d_vii_matrix_conj = conj(d_vii_matrix);
//        d_vii_matrix_trans = trans(d_vii_matrix);
    }

    /*
     * Our virtual destructor.
     */
    MUSIC_lin_array_cnx_impl::~MUSIC_lin_array_cnx_impl()
    {
      std::cout << "Total output items produced: " << nout_items_total << std::endl;

      delete connex;
      free(in0_i);
      free(in1_i);
      free(res_mult);
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
        cx_fmat in_matrix(in + item * d_num_ant_ele * d_num_ant_ele, d_num_ant_ele, d_num_ant_ele);
        fvec out_vec(out + item * d_pspectrum_len, d_pspectrum_len, COPY_MEM, FIX_SIZE);

        // determine EVD of the auto-correlation matrix
        eig_sym(eig_val, eig_vec, in_matrix);

        // noise subspace and its square matrix
        U_N = eig_vec.cols(0, d_num_ant_ele - d_num_targets - 1);

        U_N_sq = U_N*trans(U_N);

        /*====================================================================
         * Determine pseudo-spectrum for each value of theta in [0.0, 180.0)
         *===================================================================*/

        // We have a number of d_spectrum_len arrays to be multiplied by the
        // same matrix U_N_sq and we process the matrix multiplications in chunks

        // Pointers to the current and the next input chunks for the CnxArr
        uint16_t *arr_curr_cnx, *arr_next_cnx, *mat_cnx;

        // Prepare the first chunk
        arr_curr_cnx = in0_i;
        mat_cnx = in1_i;

        // Pointers to the current and past result chunks
        int32_t *res_curr_cnx = NULL, *res_past_cnx = NULL;

        // Indices of next, past and current array chunks in matrix format
        int idx_curr_chunk = 0, idx_next_chunk, idx_past_chunk;

        // Prepare current array and matrix for storage in Connex
        prepareInArrConnex(arr_curr_cnx, d_vii_matrix_conj, arr_per_chunk, idx_curr_chunk);
        prepareInMatConnex(mat_cnx, U_N_sq);

        connex->writeDataToArray(mat_cnx, 1, 900);

        int last_chunk = nr_chunks - 1;
        int ls_to_write = 0;

        executeLocalKernel(connex, "initIndex");

        for (int cnt_chunk = 0; cnt_chunk < nr_chunks; cnt_chunk++) {
          res_curr_cnx = &res_mult[cnt_chunk * nr_elem_calc_c];

          connex->writeDataToArray(arr_curr_cnx, process_at_once, ls_to_write);

          int res = executeLocalKernel(connex, "multiplyArrMatKernel");
          if (res) {
            return noutput_items;
          }

          // Prepare future data for all but the last chunk
          if (cnt_chunk != last_chunk) {
            arr_next_cnx = arr_curr_cnx + process_at_once * vector_array_size;
            idx_next_chunk = idx_curr_chunk + arr_per_chunk;

            prepareInArrConnex(arr_next_cnx, d_vii_matrix_conj, arr_per_chunk, idx_next_chunk);
          }

          // Process past data for all but the first chunk
          if (cnt_chunk != 0) {
            int idx_past_results = (cnt_chunk - 1) * arr_per_chunk;

            prepareOutDataConnex(res_temp, res_past_cnx);
            processOutData(out_vec, idx_past_results, res_temp, d_vii_matrix, idx_past_chunk);
          }

          // 2 * ---> complex elements, so we have real *and* imag parts
          connex->readMultiReduction(nr_elem_calc_c, res_curr_cnx);

          // Increment for next chunk
          arr_curr_cnx = arr_next_cnx;
          idx_past_chunk = idx_curr_chunk;
          idx_curr_chunk = idx_next_chunk;
          res_past_cnx = res_curr_cnx;
          ls_to_write += process_at_once;
        } // end loop for each chunk

        // Results for the last chunk
        prepareOutDataConnex(res_temp, res_past_cnx);
        processOutData(out_vec, last_chunk * arr_per_chunk, res_temp, d_vii_matrix, idx_past_chunk);

        out_vec = 10.0 * log10(out_vec/out_vec.max());

      }

      nout_items_total += noutput_items;

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

    /*===================================================================
     * Method that prepare in/out data to work with the ConnexArray
     * Prepare = scale and cast
     *===================================================================*/
    // TODO possibly pointless to pass arr_to_prepare
    void MUSIC_lin_array_cnx_impl::prepareInArrConnex(
      uint16_t *out_arr, const cx_fmat &in_data, const int arr_to_prepare,
      const int arr_to_start)
    {
      // in_data is a complex matrix whose columns represent an array to be
      // multiplied with the matrix and has a number of pspectrum_len columns of
      // such arrays. Therefore, it's a matrix of size (d_num_ant_ele x
      // pspectrum_len), aka (arr_size x nr_arrays)

      int idx_cnx = 0;

      for (int j = arr_to_start; j < arr_to_start + arr_to_prepare; j++) { // iterate through arrays
        // Each array has to be multiplied with each column of the matrix, so we
        // store each array a number of arr_size consecutively
        for (int k = 0; k < arr_size; k++) {
          for (int i = 0; i < arr_size; i++) { // iterate through elements of array
            out_arr[idx_cnx++] = static_cast<uint16_t>(real(in_data(i, j)) * factor_mult1);
            out_arr[idx_cnx++] = static_cast<uint16_t>(imag(in_data(i, j)) * factor_mult1);
          }
        }
      }
    }

    void MUSIC_lin_array_cnx_impl::prepareInMatConnex(
      uint16_t *out_mat, const cx_fmat &in_mat)
    {
      // Only one LS will contain the matrix => See how many times we have to
      // repeat it to store it in the one LS
      const int nr_repeats = vector_array_size / mat_size_c;
      int idx_cnx = 0;

      for (int cnt_r = 0; cnt_r < nr_repeats; cnt_r++) {
        // Store column-first
        for (int j = 0; j < arr_size; j++) {
          for (int i = 0; i < arr_size; i++) {
            out_mat[idx_cnx++] = static_cast<uint16_t>(real(in_mat(i, j)) * factor_mult2);
            out_mat[idx_cnx++] = static_cast<uint16_t>(imag(in_mat(i, j)) * factor_mult2);
          }
        }
      }
    }

    void MUSIC_lin_array_cnx_impl::prepareOutDataConnex(
      cx_fmat &out_data, const int32_t *raw_out_data)
    {
      float temp0, temp1;
      int cnt_cnx = 0;

      // Resultig array of a multiplication is stored column-wise for faster
      // access, since Armadillo matrices are stored column-wise
      for (int i = 0; i < arr_per_chunk; i++) {
        for (int j = 0; j < arr_size; j++) {
          temp0 = (static_cast<float>(raw_out_data[cnt_cnx++]));
          temp1 = (static_cast<float>(raw_out_data[cnt_cnx++]));

          out_data(i, j) = gr_complex(temp0, temp1);
        }
      }
    }

    void MUSIC_lin_array_cnx_impl::processOutData(
      fvec &out_vec, const int idx_to_start, cx_fmat &temp_res, cx_fmat &in_arr,
      const int arr_to_start)
    {
      int idx_out = idx_to_start;
      gr_complex temp_out;

      int j, k;

      for (j = 0, k = arr_to_start; j < arr_per_chunk; j++, k++) {
        temp_out = as_scalar(temp_res.row(j) * in_arr.col(k));
        out_vec(idx_out++) = 1.0 / (temp_out.real() / factor_res);
      }
    }

    /*===================================================================
     * Define ConnexArray kernels that will be used in the worker
     *===================================================================*/
    int MUSIC_lin_array_cnx_impl::executeLocalKernel(ConnexMachine *connex,
      std::string kernel_name)
    {
      try {
        connex->executeKernel(kernel_name.c_str());
      } catch (std::string e) {
        std::cout << e << std::endl;
        return -1;
      }
      return 0;
    }


    void MUSIC_lin_array_cnx_impl::init_kernel(int size_of_block)
    {
      BEGIN_KERNEL("initKernel");
        EXECUTE_IN_ALL(
          R25 = 0;
          R26 = 900;
          R30 = 1;
          R31 = 0;
          R28 = size_of_block;  // Equal to ARR_SIZE_C; dimension of the blocks
                                // on which reduction is performed at once
          R9 = INDEX;           // Select only the odd PEs
        )
      END_KERNEL("initKernel");
    }

    void MUSIC_lin_array_cnx_impl::init_index(void)
    {
      BEGIN_KERNEL("initIndex");
        EXECUTE_IN_ALL(
          R25 = 0;
          R2 = LS[R26];           // load input matrix
        )
      END_KERNEL("initIndex");
    }

    void MUSIC_lin_array_cnx_impl::multiply_kernel(
      int process_at_once, int size_of_block, int blocks_to_reduce)
    {
      BEGIN_KERNEL("multiplyArrMatKernel");
        for (int i = 0; i < process_at_once; i++) {
          EXECUTE_IN_ALL(
            R1 = LS[R25];         // load input array
            R29 = INDEX;          // Used later to select PEs for reduction
            R27 = size_of_block;  // Used to select blocks for reduction

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

            R10 = R9 & R30;
            R7 = (R10 == R30);
          )

          EXECUTE_WHERE_EQ(       // Only in the odd PEs
            // Getting -b1 * b2 in each odd cell
            R3 = R31 - R3;        // All partial real parts are in R3

            R4 = R5;              // All partial imaginary parts are now in R4
          )

          REPEAT_X_TIMES(blocks_to_reduce);
            EXECUTE_IN_ALL(
              R7 = (R29 < R27);   // Select only blocks of PEs at a time
            )
            EXECUTE_WHERE_LT(
              R29 = 129;          // A random number > 128 so these PEs won't be
                                  // selected again
              REDUCE(R3);         // Real part
              REDUCE(R4);         // Imaginary part
            )
            EXECUTE_IN_ALL(
              R27 = R27 + R28;    // Go to the next block of PEs
            )
          END_REPEAT;

          EXECUTE_IN_ALL(
            R25 = R25 + R30;      // Go to the next LS
          )
        }
      END_KERNEL("multiplyArrMatKernel");
    }

  } /* namespace doa */
} /* namespace gr */

