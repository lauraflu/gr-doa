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

#include "ConnexMachine.h"

#define MIN(x, y) ((x > y) ? y : x)

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
      cx_fmat d_vii_matrix_conj;

      cx_fmat res_temp;

      /*====================================================================
       * CONNEX KERNEL RELATED
       *===================================================================*/

      int nout_items_total = 0;

      ConnexMachine *connex;
      std::string init_kernel_name;
      std::string mult_kernel_name;

      // Variables for easier management of chunks and sizes
      const int vector_array_size = 128;
      const int local_storage_size = 1024;

      int arr_size, mat_size;
      int arr_size_c, mat_size_c;

      // Nr of times the *same* mat/arr is repeated in a LS
      int nr_repeat_mat, nr_repeat_arr;

      // In case the data doesn't fill the whole ConnexArray
      int padding;

      // Nr of *different* arrays/columns of matrix that fit in a LS
      int arr_per_LS, mat_cols_per_LS;

      // Nr of arrays are processed in a chunk
      int arr_per_chunk;
      int nr_chunks;
      int LS_per_mat;
      int LS_per_chunk;

      // Nr of reductions to read in a chunk
      int red_per_chunk;

      // The total number of the arrays that will be multiplied by the same
      // matrix
      int nr_arrays;

      // Pointers to data for/from the ConnexArray
      uint16_t *in0_i, *in1_i;
      int32_t *res_mult;

      // Factors for scaling the input data for the ConnexArray
      int factor_mult1, factor_mult2, factor_res;

      // Executes the kernel
      int executeLocalKernel(ConnexMachine *connex, std::string kernel_name);

      // Defines the init kernel
      void init_kernel(int size_of_block);

      // Kernels for processing with small data
      // Defines the init kernel
      void init_index(void);

      // Defines the processing kernel
      void multiply_kernel(
        int LS_per_iteration,
        int size_reduction_block,
        int blocks_to_reduce);

      // Kernels for processing with large data
      void init_index_large(void);

      void multiply_kernel_large(
        int LS_per_iteration,
        int LS_per_mat,
        int size_reduction_block,
        int blocks_to_reduce,
        int blocks_to_reduce_last);


      /* \brief Prepares (scales and converts to uint16_t) the elements of an
       *        array which will be stored in the ConnexArray.
       * \param out_arr Pointer to the block of data that will be fed to the
       *        ConnexArray. The size of this block needs to be at least
       *        (arr_to_prepare * arr_size * arr_size * 2). The 2 comes from the
       *        fact that the elements to prepare are complex, so we need space
       *        for both the real and the imaginary part.
       * \param in_data The input matrix that contains elements to be prepared.
       *        Each column is an array that will be multiplied with a matrix in
       *        the kernel. It needs to have at least (arr_to_prepare * arr_size
       *        * arr_size) elements.
       * \param arr_to_prepare The number of arrays in a chunk that will be
       *        prepared
       * \param arr_to_start The index of the array from which to start
       */
      void prepareInArrConnex(
        uint16_t *out_arr,
        const cx_fmat &in_data);


      /* \brief Prepares (scales and converts) the elements of the matrix that
       *        will be fed to the ConnexArray. For each output item, the matrix
       *        that is multiplied with the arrays is the same, so it will
       *        spread across only a LS.
       * \param out_mat Pointer to the block of data that will be fed to the
       *        ConnexArray. Needs to have space for at least vector_array_size
       *        elements.
       * \param in_mat The matrix that needs to be prepared. Needs to have at
       *        least arr_size * arr_size elements.
       */
      void prepareInMatConnex(uint16_t *out_mat, const cx_fmat &in_mat);

      void prepareOutDataConnex(cx_fmat &out_data, const int32_t *raw_out_data);

      void processOutData(
        fvec &out_vec,
        const int idx_to_start,
        cx_fmat &temp_res,
        cx_fmat &in_arr,
        const int arr_to_start);

    void splitArraysInChunks(int &arr_per_chunk_, int &nr_chunks_,
      const int &LS_for_mat_, const int &nr_arrays_, const int &arr_per_LS_);

    void calculateChunkingParameters(
      int &nr_red_blocks, int &size_red_block, int &nr_red_last_mat_chunk);

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

