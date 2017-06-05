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

    std::vector<uint16_t> row_nr;
    std::vector<uint16_t> col_nr;
    std::vector<gr_complex> out_data(16);

    void executeAutocorrelationKernel(ConnexMachine *connex);
    void autocorrelationKernel(const int n_rows_, const int n_cols_, const int nr_loops);

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
      , n_rows(d_num_inputs)
      , n_cols(d_snapshot_size)
    {
      try {
        connex = new ConnexMachine(distributionFIFO,
                                   reductionFIFO,
                                   writeFIFO,
                                   readFIFO);
      } catch (std::string err) {
        std::cout << err << std::endl;
      }

      factor_mult = 1 << 14;
      factor_res = 1 << 12;

      const int nr_loops = (n_cols * 2) / vector_array_size;

      try {
        autocorrelationKernel(n_rows, n_cols, nr_loops);
      } catch (std::string err) {
        std::cout << err << std::endl;
      }

      n_elems = n_rows * n_cols;
      n_elems_c = 2 * n_elems;
      n_elems_out = n_rows * n_rows;
      n_elems_out_c = n_elems_out * 2;
      n_ls_busy = n_elems_c / vector_array_size;
      // How many reductions will be performed on Connex
      n_red_per_elem = ((n_cols * 2) / vector_array_size) * 2;

      in_data_cnx =
        static_cast<uint16_t *>(malloc(n_elems_c * sizeof(uint16_t)));
      out_data_cnx =
        static_cast<int32_t *>(malloc(n_red_per_elem * sizeof(int32_t)));

      d_nonoverlap_size = d_snapshot_size-d_overlap_size;
      set_history(d_overlap_size+1);

      // Create container for temporary matrix
      d_input_matrix = arma::cx_fmat(snapshot_size,inputs);

      // initialize the reflection matrix
      d_J.eye(d_num_inputs, d_num_inputs);
      d_J = fliplr(d_J);
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
//        gr_complex *out_data = &out[i * d_num_inputs * d_num_inputs];
        std::vector<const gr_complex *> in_data_ptr(d_num_inputs);

        // Keep pointers to input data
        for (int k = 0; k < d_num_inputs; k++) {
          in_data_ptr[k] = ((gr_complex *)input_items[k] + i * d_nonoverlap_size);
          prepareInData(&in_data_cnx[k * 2 * n_cols], in_data_ptr[k], n_cols);
        }

        connex->writeDataToArray(in_data_cnx, n_ls_busy, 0);

        for (int cnt_row = 0; cnt_row < n_rows; cnt_row++) {
          // Only elements higher or equal than the main diagonal
          for (int cnt_col = cnt_row; cnt_col < n_rows; cnt_col++) {
//            std::cout << "row: " << cnt_row << ", col: " << cnt_col << std::endl;

            // TODO maybe better with real assignation; this destroys and
            // creates new elements with this value
            row_nr.assign(vector_array_size, cnt_row);
            col_nr.assign(vector_array_size, cnt_col);

            connex->writeDataToArray(row_nr.data(), 1, 1022);
            connex->writeDataToArray(col_nr.data(), 1, 1023);

            executeAutocorrelationKernel(connex);

            connex->readMultiReduction(n_red_per_elem, out_data_cnx);

            // process out data; consider out array stored column-first
            int curr_idx = cnt_row + cnt_col * n_rows;
            out_data[curr_idx] =
              prepareAndProcessOutData(out_data_cnx, n_red_per_elem);
            // Hermitian matrix => a_ij = conj(a_ji);
            out_data[cnt_col + cnt_row * n_rows] =
              gr_complex(out_data[curr_idx].real(), -out_data[curr_idx].imag());

            std::cout << "data_out[" << cnt_row << ", " << cnt_col << "] = " <<
              "data_out[" << cnt_col << ", " << cnt_row << "] = " <<
              out_data[curr_idx] << std::endl;
          }
        }

        // Form input matrix
        for(int k=0; k<d_num_inputs; k++)
        {
            memcpy((void*)d_input_matrix.colptr(k),
            ((gr_complex*)input_items[k] + i * d_nonoverlap_size),
            sizeof(gr_complex) * d_snapshot_size);

//            std::cout << "Check Input: " << std::endl;
//            std::cout << "exp: " << d_input_matrix[0] << ", got: " <<
//            in_data_cnx
        }
        // Make output pointer into matrix pointer
        arma::cx_fmat out_matrix(
          out + d_num_inputs * d_num_inputs * i,
          d_num_inputs, d_num_inputs, COPY_MEM, FIX_SIZE);


        // Do autocorrelation
        out_matrix =
          (1.0 / d_snapshot_size) * d_input_matrix.st() * conj(d_input_matrix);
//        if (d_avg_method == 1)
//            out_matrix = 0.5 * out_matrix +
//              (0.5/d_snapshot_size) * d_J * conj(out_matrix) * d_J;

        for (int k = 0; k < n_rows * n_rows; k++) {
          std::cout << "Exp: " << out_matrix[k] << ", got: " << out_data[k] <<
          std::endl;
        }

      }

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

    void autocorrelateConnexKernel_impl::prepareOutData(
      gr_complex *out_data, const int32_t *in_data, const int n_elems_in)
    {
      float temp_real, temp_imag;
      for (int i = 0; i < n_elems_in; i+=2) {
        temp_real = static_cast<float>(in_data[i]) / factor_res;
        temp_imag = static_cast<float>(in_data[i + 1]) / factor_res;

        out_data[i / 2] = gr_complex(temp_real, temp_imag);
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
//        std::cout << "out_data[" << i / 2 << "] = " << out_data << std::endl;
      }
    }

    gr_complex autocorrelateConnexKernel_impl::prepareAndProcessOutData(
      const int32_t *in_data, const int n_elems_in)
    {
      float temp_real, temp_imag;
      float acc_real = 0, acc_imag = 0;
      for (int i = 0; i < n_elems_in; i+=2) {
        temp_real = static_cast<float>(in_data[i]) / factor_res;
        temp_imag = static_cast<float>(in_data[i + 1]) / factor_res;

        acc_real += temp_real;
        acc_imag += temp_imag;

//        std::cout << gr_complex(temp_real, temp_imag) << std::endl;
      }

      // Divide by the snapshot size = n_cols
      acc_real = acc_real / n_cols;
      acc_imag = acc_imag / n_cols;

      return gr_complex(acc_real, acc_imag);
    }

    void executeAutocorrelationKernel(ConnexMachine *connex)
    {
      connex->executeKernel("autocorrelation");
    }

    void autocorrelationKernel(
      const int n_rows_,
      const int n_cols_,
      const int nr_loops)
    {
      BEGIN_KERNEL("autocorrelation");
        EXECUTE_IN_ALL(
          // Some constants
          R0 = nr_loops;
          R29 = 1;
          R28 = 0;

          // TODO what's the right way to put the indices in LS to be read?
          R30 = LS[1022];         // line i
          R31 = LS[1023];         // col j

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
            LS[500] = R4;
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
      END_KERNEL("autocorrelation");
    }


  } /* namespace doa */
} /* namespace gr */

