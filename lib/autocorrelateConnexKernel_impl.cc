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
#include "autocorrelateConnexKernel_impl.h"
#include <armadillo>
#define COPY_MEM false  // Do not copy matrices into separate memory
#define FIX_SIZE true   // Keep dimensions of matrices constant

namespace gr {
  namespace doa {


    int executeLocalKernel(ConnexMachine *connex, std::string kernel_name);
    void autocorrelationKernel(const int nr_loops);
    void initKernel(const int nr_loops);
    void initIndex();
    void multiplyKernel(void);

    autocorrelateConnexKernel::sptr
    autocorrelateConnexKernel::make(
      int inputs,
      int snapshot_size,
      int overlap_size,
      int avg_method,
      std::string distributionFIFO,
      std::string reductionFIFO,
      std::string writeFIFO,
      std::string readFIFO)
    {
      return gnuradio::get_initial_sptr
        (new autocorrelateConnexKernel_impl(inputs, snapshot_size, overlap_size, avg_method,
          distributionFIFO, reductionFIFO, writeFIFO, readFIFO));
    }

    /*
     * The private constructor
     */
    autocorrelateConnexKernel_impl::autocorrelateConnexKernel_impl(
      int inputs,
      int snapshot_size,
      int overlap_size,
      int avg_method,
      std::string distributionFIFO,
      std::string reductionFIFO,
      std::string writeFIFO,
      std::string readFIFO)
      : gr::block("autocorrelateConnexKernel",
              gr::io_signature::make(inputs, inputs, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex)*inputs*inputs))
      , d_num_inputs(inputs)
      , d_snapshot_size(snapshot_size)
      , d_overlap_size(overlap_size)
      , d_avg_method(avg_method)
    {
      try {
        connex = new ConnexMachine(distributionFIFO,
                                   reductionFIFO,
                                   writeFIFO,
                                   readFIFO);
      } catch (std::string err) {
        std::cout << err << std::endl;
      }

      n_rows = d_num_inputs;
      n_cols = d_snapshot_size;

      factor_mult = 1 << 13;
      factor_res = 1 << 10;

      n_elems = n_rows * n_cols;
      n_elems_c = 2 * n_elems;
      n_elems_out = n_rows * n_rows;
      n_elems_out_c = n_elems_out * 2;
      n_ls_busy = n_elems_c / vector_array_size;

      // Divide the number of complex elements in a col by the vector array size
      const int nr_loops = (n_cols * 2) / vector_array_size;
      // How many reductions will be performed on Connex = 2 per loop
      n_red_per_elem = nr_loops * 2;


      try {
        autocorrelationKernel(nr_loops);
        initKernel(nr_loops);
        initIndex();
      } catch (std::string err) {
        // This should normally not continue, but since we're unit testing are
        // recreating the same modules before closing the simulator, we have to
        // do it.
        std::cout << err << std::endl;
      }

      executeLocalKernel(connex, "initKernel");

      int nr_elem_calc = (n_rows * (n_rows + 1)) / 2;

      in_data_cnx = static_cast<uint16_t *>(malloc(n_elems_c * sizeof(uint16_t)));
      out_data_cnx = static_cast<int32_t *>(malloc(nr_elem_calc * n_red_per_elem * sizeof(int32_t)));

      idx_val.resize(n_rows);
      for (int i = 0; i < n_rows; i++) {
        idx_val[i].resize(vector_array_size);
        for (int j = 0; j < vector_array_size; j++) {
          idx_val[i][j] = i;
        }
      }

      int ls_idx = 900;
      // Pre-load index in array
      for (int cnt_row = 0; cnt_row < n_rows; cnt_row++) {
        for (int cnt_col = cnt_row; cnt_col < n_rows; cnt_col++) {
            connex->writeDataToArray(idx_val[cnt_row].data(), 1, ls_idx++);
            connex->writeDataToArray(idx_val[cnt_col].data(), 1, ls_idx++);
        }
      }

      d_nonoverlap_size = d_snapshot_size - d_overlap_size;
      set_history(d_overlap_size + 1);

      if (d_avg_method) {
        refl_matrix.resize(n_rows);
        for (int i = 0; i < n_rows; i++) {
          refl_matrix[i].resize(n_rows);
        }
      }
    }

    /*
     * Our virtual destructor.
     */
    autocorrelateConnexKernel_impl::~autocorrelateConnexKernel_impl()
    {
      delete connex;
      free(in_data_cnx);
      free(out_data_cnx);
    }

    void
    autocorrelateConnexKernel_impl::forecast(
      int noutput_items,
      gr_vector_int &ninput_items_required)
    {
      // Setup input output relationship
      for (int i=0; i<ninput_items_required.size(); i++)
        ninput_items_required[i] = d_nonoverlap_size*noutput_items;
    }

    int
    autocorrelateConnexKernel_impl::general_work (int output_matrices,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      // Cast pointer
      gr_complex *out = (gr_complex *) output_items[0];

      // Create each output matrix
      for (int i = 0; i < output_matrices; i++)
      {
        gr_complex *out_data = &out[i * d_num_inputs * d_num_inputs];
        std::vector<const gr_complex *> in_data_ptr(d_num_inputs);

        // Keep pointers to input data
        for (int k = 0; k < d_num_inputs; k++) {
          in_data_ptr[k] = ((gr_complex *)input_items[k] + i * d_nonoverlap_size);
          prepareInData(&in_data_cnx[k * 2 * n_cols], in_data_ptr[k], n_cols);
        }

        // Re-initialize index value for each output item
        executeLocalKernel(connex, "initIndex");
        connex->writeDataToArray(in_data_cnx, n_ls_busy, 0);

        int32_t *curr_out_data_cnx = out_data_cnx, *past_out_data_cnx;
        int past_row = 0, past_col = 0;
        int past_idx, sym_idx;

        for (int cnt_row = 0; cnt_row < n_rows; cnt_row++) {
          // Only elements higher or equal than the main diagonal
          for (int cnt_col = cnt_row; cnt_col < n_rows; cnt_col++) {

            executeLocalKernel(connex, "autocorrelationKernel");

            // Process past data for all but the first element
            if (!(cnt_row == 0 && cnt_col == 0)) {

              // Output array stored column-first
              past_idx = past_row + past_col * n_rows;
              sym_idx = past_col + past_row * n_rows;
              out_data[past_idx] =
                prepareAndProcessOutData(past_out_data_cnx, n_red_per_elem);

              // Hermitian matrix => a_ij = conj(a_ji);
                out_data[sym_idx] =
                  gr_complex(out_data[past_idx].real(), -out_data[past_idx].imag());
            }

            connex->readMultiReduction(n_red_per_elem, curr_out_data_cnx);

            past_out_data_cnx = curr_out_data_cnx;
            curr_out_data_cnx += n_red_per_elem;
            past_row = cnt_row;
            past_col = cnt_col;
          }
        }

        // Process data for the last chunk
        past_idx = (n_rows - 1) + (n_rows - 1) * n_rows;
        out_data[past_idx] = prepareAndProcessOutData(past_out_data_cnx, n_red_per_elem);

        // Averaging results
        // TODO: check if it's faster to use arma here
        if (d_avg_method) {
          std::complex<float> two_c = (2.0, 2.0);

          for (int cnt_row = 0; cnt_row < n_rows; cnt_row++) {

            for (int cnt_col = 0; cnt_col < n_rows; cnt_col++) {
              int idx_row = n_rows - 1 - cnt_row;
              int idx_col = n_rows - 1 - cnt_col;
              int idx = idx_row + idx_col * n_rows;

              // Divide the initial results by 2
              out_data[idx] = out_data[idx] / two_c;

              // form reflection matrix
              refl_matrix[cnt_row][cnt_col] = conj(out_data[idx]);
            }
          }

          for (int cnt_row = 0; cnt_row < n_rows; cnt_row++) {
            for (int cnt_col = 0; cnt_col < n_rows; cnt_col++) {
              int idx = cnt_row + cnt_col * n_rows;
              out_data[idx] += refl_matrix[cnt_row][cnt_col];
            }
          }
        } // end loop averaging
      } // end loop for each output matrix

      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (d_nonoverlap_size*output_matrices);

      // Tell runtime system how many output items we produced.
      return (output_matrices);
    }

    void autocorrelateConnexKernel_impl::prepareInData(
      uint16_t *out_data, const gr_complex *in_data, const int n_elems_in)
    {
      for (int i = 0; i < n_elems_in; i++) {
        out_data[2 * i] = static_cast<uint16_t>(in_data[i].real() * factor_mult);
        out_data[2 * i + 1] = static_cast<uint16_t>(in_data[i].imag() * factor_mult);
      }
    }

    void autocorrelateConnexKernel_impl::printOutData(
      const uint16_t *in_data, const int n_elems_in)
    {
      float temp_real, temp_imag;
      for (int i = 0; i < n_elems_in; i+=2) {
        temp_real = (static_cast<float>(in_data[i])) / factor_mult;
        temp_imag = (static_cast<float>(in_data[i + 1])) / factor_mult;

        gr_complex out_data(temp_real, temp_imag);
        std::cout << "out_data[" << i / 2 << "] = " << out_data << std::endl;
      }
    }

    gr_complex autocorrelateConnexKernel_impl::prepareAndProcessOutData(
      const int32_t *in_data, const int n_elems_in)
    {
      float temp_real, temp_imag;
      float acc_real = 0, acc_imag = 0;
      for (int i = 0; i < n_elems_in; i+=2) {
        temp_real = static_cast<float>(in_data[i]);
        temp_imag = static_cast<float>(in_data[i + 1]);

        acc_real += temp_real;
        acc_imag += temp_imag;
      }

      // Divide by the snapshot size = n_cols
      acc_real = (acc_real / factor_res) / n_cols;
      acc_imag = (acc_imag / factor_res) / n_cols;

      return gr_complex(acc_real, acc_imag);
    }

    int executeLocalKernel(ConnexMachine *connex, std::string kernel_name)
    {
      try {
        connex->executeKernel(kernel_name.c_str());
      } catch(std::string e) {
        std:: cout << "Exception in kernel execution!" << std::endl;
        std::cout << e << std::endl;
        return -1;
      }
      return 0;
    }

    void initKernel(const int nr_loops) {
      BEGIN_KERNEL("initKernel");
        EXECUTE_IN_ALL(
          // Some constants
          R0 = nr_loops;
          R29 = 1;
          R28 = 0;
          R25 = 2;
        )
      END_KERNEL("initKernel");
    }

    void initIndex(void) {
      BEGIN_KERNEL("initIndex");
        EXECUTE_IN_ALL(
          R26 = 900;            // From here are loaded the indices for the line
          R27 = 901;            // and the column
        )
      END_KERNEL("initIndex");
    }

    void autocorrelationKernel(const int nr_loops)
    {
      BEGIN_KERNEL("autocorrelationKernel");
        EXECUTE_IN_ALL(
          // TODO what's the right way to put the indices in LS to be read?
          R30 = LS[R26];         // line i
          R31 = LS[R27];         // col j

          // keep indices from where to load the lines and columns
          R30 = R30 * R0;         // go to the LS with line i
          R31 = R31 * R0;         // go to the Ls with col j
        )

        REPEAT_X_TIMES(nr_loops);
          EXECUTE_IN_ALL(
            R1 = LS[R30];
            R2 = LS[R31];

            R3 = R1 * R2;
            R3 = MULT_HIGH();     // re1 * re2, im1 * im2

            CELL_SHL(R2, R29);
            NOP;
            R4 = SHIFT_REG;
            R4 = R4 * R1;
            R4 = MULT_HIGH();     // re1 * im2
            R4 = R28 - R4;        // The column is conjugated => negate these

            CELL_SHR(R2, R29);
            NOP;
            R5 = SHIFT_REG;
            R5 = R5 * R1;
            R5 = MULT_HIGH();     // re2 * im1;

            R6 = INDEX;           // Select only the odd PEs
            R6 = R6 & R29;
            R7 = (R6 == R29);
          )

          EXECUTE_WHERE_EQ(
            R4 = R5;              // All partial imaginary parts are now in R4
          )

          EXECUTE_IN_ALL(
            REDUCE(R3);
            REDUCE(R4);

            R30 = R30 + R29;      // Go to next chunk on line & col
            R31 = R31 + R29;
          )
        END_REPEAT;

        EXECUTE_IN_ALL(           // Prepare index for the next chunk
          R26 = R26 + R25;        // += 2
          R27 = R27 + R25;
        )
      END_KERNEL("autocorrelationKernel");
    }

    void multiplyKernel(void)
    {
      BEGIN_KERNEL("mult");
      R3 = R1 * R2;
      R3 = MULT_HIGH();     // re1 * re2, im1 * im2

      CELL_SHL(R2, R29);
      NOP;
      R4 = SHIFT_REG;
      R4 = R4 * R1;
      R4 = MULT_HIGH();     // re1 * im2
      R4 = R28 - R4;        // The column is conjugated => negate these

      CELL_SHR(R2, R29);
      NOP;
      R5 = SHIFT_REG;
      R5 = R5 * R1;
      R5 = MULT_HIGH();     // re2 * im1;
      END_KERNEL("mult");
    }
  } /* namespace doa */
} /* namespace gr */

