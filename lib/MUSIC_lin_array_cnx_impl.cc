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
        factor_res = 1 << 14;

        // Kernel parameters
        int nr_red_blocks, size_red_block, nr_red_blocks_last;

        calculateChunkingParameters(
          nr_red_blocks,
          size_red_block,
          nr_red_blocks_last);

        std::cout << "Parameters: " << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "nr_chunks: " << nr_chunks << std::endl;
        std::cout << "arr_per_chunk: " << arr_per_chunk << std::endl;
        std::cout << "LS_per_mat: " << LS_per_mat << std::endl;
        std::cout << "LS_per_chunk: " << LS_per_chunk << std::endl;
        std::cout << "arr_per_LS: " << arr_per_LS << std::endl;
        std::cout << "mat_cols_per_LS: " << mat_cols_per_LS << std::endl;
        std::cout << "nr_repeat_arr: " << nr_repeat_arr << std::endl;
        std::cout << "nr_repeat_mat: " << nr_repeat_mat << std::endl;

        std::cout << "red_per_chunk: " << red_per_chunk << std::endl;
        std::cout << "nr_red_blocks: " << nr_red_blocks << std::endl;
        std::cout << "size_red_block: " << size_red_block << std::endl;

        // Create the kernel
        try {
          init_kernel(size_red_block);

          if (mat_size_c <= vector_array_size) {
            init_index();
            multiply_kernel(LS_per_chunk, size_red_block, nr_red_blocks);
          } else {
            init_index_large();
            multiply_kernel_large(LS_per_chunk, LS_per_mat, size_red_block,
            nr_red_blocks, nr_red_blocks_last);
          }
        } catch (std::string e) {
          std::cout << e << std::endl;
        }

        executeLocalKernel(connex, init_kernel_name.c_str());

        int size_of_padding = (nr_arrays / arr_per_LS) * padding;
        // Allocate memory for the data that will be passed to the ConnexArray
        // and the data that it produces.
        in0_i = static_cast<uint16_t *>
            (malloc((nr_arrays * arr_size_c * arr_size + size_of_padding) * sizeof(uint16_t)));
        in1_i = static_cast<uint16_t *>
            (malloc(LS_per_mat * vector_array_size * sizeof(uint16_t)));
        res_mult = static_cast<int32_t *>
            (malloc(nr_chunks * red_per_chunk * sizeof(int32_t)));

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

        // Prepare steering vectors for storage on ConnexArray
        prepareInArrConnex(in0_i, d_vii_matrix_conj);
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
        uint16_t *arr_curr_cnx = in0_i, *mat_cnx = in1_i;
        int32_t *res_curr_cnx = NULL;

        // Indices of next, past and current array chunks in matrix format
        int idx_curr_chunk = 0;

        // Prepare & write matrix for storage in Connex
        prepareInMatConnex(mat_cnx, U_N_sq);
        connex->writeDataToArray(mat_cnx, LS_per_mat, 900);

        executeLocalKernel(connex, "initIndex");

        for (int cnt_chunk = 0; cnt_chunk < nr_chunks; cnt_chunk++) {
          res_curr_cnx = &res_mult[cnt_chunk * red_per_chunk];

          connex->writeDataToArray(arr_curr_cnx, LS_per_chunk, 0);

          int res = executeLocalKernel(connex, mult_kernel_name.c_str());

          connex->readMultiReduction(red_per_chunk, res_curr_cnx);

          prepareOutDataConnex(res_temp, res_curr_cnx);

          processOutData(out_vec, cnt_chunk * arr_per_chunk, res_temp, d_vii_matrix, idx_curr_chunk);

          // Increment for next chunk
          arr_curr_cnx += LS_per_chunk * vector_array_size;
          idx_curr_chunk += arr_per_chunk;
        } // end loop for each chunk


        out_vec = 10.0 * log10(out_vec/out_vec.max());
      }

      nout_items_total += noutput_items;

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }


    void MUSIC_lin_array_cnx_impl::splitArraysInChunks(
      int &arr_per_chunk_, int &nr_chunks_, const int &LS_for_mat_, const int
      &nr_arrays_, const int &arr_per_LS_)
    {
       // ***Find maximum possible chunk by splitting (if necessary) the total
      // number of arrays in equal chunks
      int available_LS = local_storage_size - LS_for_mat_; // only one line with the mat
      int arrays_to_fit = available_LS * arr_per_LS_;
      int remainder = 1;
      nr_chunks_ = 1;
      do {
        arr_per_chunk_ = nr_arrays_ / nr_chunks_;
        remainder = nr_arrays_ % nr_chunks_;
        nr_chunks_++;
      } while ((arr_per_chunk_ > arrays_to_fit) || // still not enough space => keep going
               (remainder != 0)); // chunk is not equal => keep going
      nr_chunks_--; // when the loop is over the nr_chunks is incremented more than needed
    }

    void MUSIC_lin_array_cnx_impl::calculateChunkingParameters(
      int &nr_red_blocks_, int &size_red_block_, int &nr_red_last_mat_chunk_)
    {
      // ***Check if we have at least one whole matrix in a LS or if the matrix is
      // split across multiple LSs
      if (mat_size_c <= vector_array_size) {
        // At least one matrix in the LS
        // Use kernels for smaller arrays
        init_kernel_name = "initKernel";
        mult_kernel_name = "multiplyArrMatKernel";

        // ***See how many whole matrices fit in a LS
        nr_repeat_mat = vector_array_size / mat_size_c;
        padding = vector_array_size - nr_repeat_mat * mat_size_c;
        nr_repeat_arr = arr_size; // repeat for each column in the array
        mat_cols_per_LS = arr_size;

        arr_per_LS = nr_repeat_mat;

        LS_per_mat = 1;

        // ***Find maximum array chunk
        splitArraysInChunks(arr_per_chunk, nr_chunks, LS_per_mat, nr_arrays, arr_per_LS);

        // ***Calculate parameters per kernel
        nr_red_blocks_ = nr_repeat_mat * arr_size; // calculated per LS
        nr_red_last_mat_chunk_ = 0;
      } else {
        // Split matrix in chunks
        // Use kernels for larger arrays
        init_kernel_name = "initKernelLarge";
        mult_kernel_name = "multiplyArrMatKernelLarge";

        // ***See how many whole matrix columns fit in a LS
        int mat_col_size = arr_size_c; // just an alias
        mat_cols_per_LS = vector_array_size / mat_col_size;
        padding = vector_array_size % mat_col_size;

        // ***See how many LSs we need to store a mat
        LS_per_mat = arr_size / mat_cols_per_LS;
        int remainder_last_mat_chunk = arr_size % mat_cols_per_LS;
        if (remainder_last_mat_chunk != 0) {
          // Last LS with matrix columns is incomplete
          LS_per_mat++;
          nr_red_last_mat_chunk_ = remainder_last_mat_chunk;
        } else {
          // Last LS with matrix columns is complete
          nr_red_last_mat_chunk_ = mat_cols_per_LS;
        }
        nr_red_blocks_ = mat_cols_per_LS; // calculated per LS

        nr_repeat_arr = mat_cols_per_LS;
        nr_repeat_mat = 0; // set to zero to differentiate from the first case

        arr_per_LS = 1;

        // ***Find maximum array chunk
        splitArraysInChunks(arr_per_chunk, nr_chunks, LS_per_mat, nr_arrays, arr_per_LS);
      }
      size_red_block_ = arr_size_c;
      LS_per_chunk = arr_per_chunk / arr_per_LS;

      // 2 * => because we have one reduction for the real part and one for the imaginary
      red_per_chunk = 2 * nr_red_blocks_ * LS_per_chunk;
    }

    /*===================================================================
     * Method that prepares in/out data to work with the ConnexArray
     * Prepare = scale and cast
     *===================================================================*/
    void MUSIC_lin_array_cnx_impl::prepareInArrConnex(
      uint16_t *out_arr, const cx_fmat &in_data)
    {
      int idx_cnx = 0;

      // TODO create separate branches for padding to avoid checking everytime
      for (int j = 0; j < nr_arrays; j++) { // for each array
        for (int k = 0; k < nr_repeat_arr; k++) { // store each array this many times
          for (int i = 0; i < arr_size; i++) {
            out_arr[idx_cnx++] = static_cast<uint16_t>(real(in_data(i, j)) * factor_mult1);
            out_arr[idx_cnx++] = static_cast<uint16_t>(imag(in_data(i, j)) * factor_mult1);
          }
        }
        if (padding != 0)
          if ((j + 1) % arr_per_LS == 0) {
            // Must add padding before the next element
            idx_cnx += padding;
          }
      }
    }

    void MUSIC_lin_array_cnx_impl::prepareInMatConnex(
      uint16_t *out_mat, const cx_fmat &in_mat)
    {
      int idx_cnx = 0;

      // Separate branches to avoid checking for padding each time
      if (nr_repeat_mat != 0) {
        // Don't need to add padding
        for (int cnt_r = 0; cnt_r < nr_repeat_mat; cnt_r++) { // repeat this many times
          for (int j = 0; j < arr_size; j++) { // for each column
            for (int i = 0; i < arr_size; i++) { // for each element in column
              out_mat[idx_cnx++] = static_cast<uint16_t>(real(in_mat(i, j)) * factor_mult2);
              out_mat[idx_cnx++] = static_cast<uint16_t>(imag(in_mat(i, j)) * factor_mult2);
            }
          }
        }
      } else {
        // Don't need to repeat the mat
        for (int j = 0; j < arr_size; j++) { // for each column
          for (int i = 0; i < arr_size; i++) { // for each element in column
            out_mat[idx_cnx++] = static_cast<uint16_t>(real(in_mat(i, j)) * factor_mult2);
            out_mat[idx_cnx++] = static_cast<uint16_t>(imag(in_mat(i, j)) * factor_mult2);
          }
          // Add pading after a number of mat_cols_per_LS columns
          if ((j + 1) % mat_cols_per_LS == 0) {
            idx_cnx += padding;
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
//          std::cout << "arr " << i << ": " << out_data(i, j) << std::endl;
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
//        for (int i = 0; i < arr_size; i++) {
//          std::cout << "array " << k << ": " << temp_res(j, i) << ", ";
//        }
//        std::cout << std::endl;
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
          R25 = 0;                      // reset array LS index
          R2 = LS[R26];                 // load input matrix
        )
      END_KERNEL("initIndex");
    }

    void MUSIC_lin_array_cnx_impl::multiply_kernel(
      int LS_per_iteration, int size_reduction_block, int blocks_to_reduce)
    {
      BEGIN_KERNEL("multiplyArrMatKernel");
        for (int i = 0; i < LS_per_iteration; i++) {
          EXECUTE_IN_ALL(
            R1 = LS[R25];               // load input array
            R29 = INDEX;                // Used later to select PEs for reduction
            R27 = size_reduction_block;        // Used to select blocks for reduction

            R3 = R1 * R2;               // a1 * a2, b1 * b2
            R3 = MULT_HIGH();

            CELL_SHL(R2, R30);          // Bring b2 to the left to calc b2 * a1
            NOP;
            R4 = SHIFT_REG;
            R4 = R1 * R4;               // a1 * b2
            R4 = MULT_HIGH();

            CELL_SHR(R2, R30);
            NOP;
            R5 = SHIFT_REG;
            R5 = R1 * R5;               // b1 * a2
            R5 = MULT_HIGH();

            R10 = R9 & R30;
            R7 = (R10 == R30);
          )

          EXECUTE_WHERE_EQ(             // Only in the odd PEs
            // Getting -b1 * b2 in each odd cell
            R3 = R31 - R3;              // All partial real parts are in R3

            R4 = R5;                    // All partial imaginary parts are now in R4
          )

          REPEAT_X_TIMES(blocks_to_reduce);
            EXECUTE_IN_ALL(
              R7 = (R29 < R27);         // Select only blocks of PEs at a time
            )
            EXECUTE_WHERE_LT(
              R29 = 129;                // A random number > 128 so these PEs won't be
                                        // selected again
              REDUCE(R3);               // Real part
              REDUCE(R4);               // Imaginary part
            )
            EXECUTE_IN_ALL(
              R27 = R27 + R28;          // Go to the next block of PEs
            )
          END_REPEAT;

          EXECUTE_IN_ALL(
            R25 = R25 + R30;            // Go to the next LS
          )
        }
      END_KERNEL("multiplyArrMatKernel");
    }


    void MUSIC_lin_array_cnx_impl::init_index_large(void)
    {
      BEGIN_KERNEL("initIndexLarge");
        EXECUTE_IN_ALL(
          R25 = 0;                      // reset array LS index
        )
      END_KERNEL("initIndexLarge");
    }

    void MUSIC_lin_array_cnx_impl::multiply_kernel_large(
      int LS_per_iteration, int LS_per_mat, int size_reduction_block, int
      blocks_to_reduce, int blocks_to_reduce_last)
    {
      BEGIN_KERNEL("multiplyArrMatKernelLarge");
        for (int i = 0; i < LS_per_iteration; i++) { // for each array in chunk
          R1 = LS[R25];                   // load input array

          // For each matrix chunk except the last
          for (int j = 0; j < LS_per_mat - 1; j++) {
            EXECUTE_IN_ALL(
              R2 = LS[R26];               // load input matrix
              R29 = INDEX;                // Used later to select PEs for reduction
              R27 = size_reduction_block; // Used to select blocks for reduction

              R3 = R1 * R2;               // a1 * a2, b1 * b2
              R3 = MULT_HIGH();

              CELL_SHL(R2, R30);          // Bring b2 to the left to calc b2 * a1
              NOP;
              R4 = SHIFT_REG;
              R4 = R1 * R4;               // a1 * b2
              R4 = MULT_HIGH();

              CELL_SHR(R2, R30);
              NOP;
              R5 = SHIFT_REG;
              R5 = R1 * R5;               // b1 * a2
              R5 = MULT_HIGH();

              R10 = R9 & R30;
              R7 = (R10 == R30);
            )

            EXECUTE_WHERE_EQ(             // Only in the odd PEs
              // Getting -b1 * b2 in each odd cell
              R3 = R31 - R3;              // All partial real parts are in R3

              R4 = R5;                    // All partial imaginary parts are now in R4
            )

            REPEAT_X_TIMES(blocks_to_reduce);
              EXECUTE_IN_ALL(
                R7 = (R29 < R27);         // Select only blocks of PEs at a time
              )
              EXECUTE_WHERE_LT(
                R29 = 129;                // A random number > 128 so these PEs won't be
                                          // selected again
                REDUCE(R3);               // Real part
                REDUCE(R4);               // Imaginary part
              )
              EXECUTE_IN_ALL(
                R27 = R27 + R28;          // Go to the next block of PEs
              )
            END_REPEAT;

            EXECUTE_IN_ALL(
              R26 = R26 + R30;            // Go to the next LS with mat
            )
          }

          // The last matrix chunk may be incomplete, so we treat it differently
          // to avoid doing unnecessary reductions
          EXECUTE_IN_ALL(
            R2 = LS[R26];               // load input matrix
            R29 = INDEX;                // Used later to select PEs for reduction
            R27 = size_reduction_block; // Used to select blocks for reduction

            R3 = R1 * R2;               // a1 * a2, b1 * b2
            R3 = MULT_HIGH();

            CELL_SHL(R2, R30);          // Bring b2 to the left to calc b2 * a1
            NOP;
            R4 = SHIFT_REG;
            R4 = R1 * R4;               // a1 * b2
            R4 = MULT_HIGH();

            CELL_SHR(R2, R30);
            NOP;
            R5 = SHIFT_REG;
            R5 = R1 * R5;               // b1 * a2
            R5 = MULT_HIGH();

            R10 = R9 & R30;
            R7 = (R10 == R30);
          )

          EXECUTE_WHERE_EQ(             // Only in the odd PEs
            // Getting -b1 * b2 in each odd cell
            R3 = R31 - R3;              // All partial real parts are in R3

            R4 = R5;                    // All partial imaginary parts are now in R4
          )

          REPEAT_X_TIMES(blocks_to_reduce_last);
            EXECUTE_IN_ALL(
              R7 = (R29 < R27);         // Select only blocks of PEs at a time
            )
            EXECUTE_WHERE_LT(
              R29 = 129;                // A random number > 128 so these PEs won't be
                                        // selected again
              REDUCE(R3);               // Real part
              REDUCE(R4);               // Imaginary part
            )
            EXECUTE_IN_ALL(
              R27 = R27 + R28;          // Go to the next block of PEs
            )
          END_REPEAT;
          // End processing for last matrix chunk

          EXECUTE_IN_ALL(
            R25 = R25 + R30;              // Go to the next array
          )
        }
      END_KERNEL("multiplyArrMatKernelLarge");
    }

  } /* namespace doa */
} /* namespace gr */

