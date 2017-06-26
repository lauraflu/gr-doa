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

    bool red_finish = false;

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

      factor_mult = 1 << 13;
      factor_res = 1 << 10;

      n_elems = n_rows * n_cols;
      n_elems_c = 2 * n_elems;
      n_elems_out = n_rows * n_rows;
      n_elems_out_c = n_elems_out * 2;

      int n_cols_c = n_cols * 2;

      if (n_cols_c < vector_array_size) {
        LS_per_row = 1;
        padding = vector_array_size - n_cols_c;
      } else {
        LS_per_row = n_cols_c / vector_array_size;
        padding = n_cols_c % vector_array_size;
        if (padding != 0)
          LS_per_row++;
      }

      total_LS_used = LS_per_row * n_rows;
      n_red_per_elem = LS_per_row * 2;

      // Number of output elements computed: n_rows * (n_rows + 1) / 2
      n_red =  (n_rows * (n_rows + 1) / 2) * n_red_per_elem;

      if (local_storage_size < total_LS_used) {
        std::cout << "Chunking not yet implemented! More input data that can be processed!"
        << std::endl;
        return;
      }

      try {
        initKernel(LS_per_row);
        autocorrelationKernel(LS_per_row);
      } catch (std::string err) {
        std::cout << err << std::endl;
      }

      // Initialize Connex registers with some values
      executeLocalKernel(connex, "initKernel");

      in_data_cnx = static_cast<uint16_t *>(malloc(n_elems_c * sizeof(uint16_t)));
      out_data_cnx = static_cast<int32_t *>(malloc(n_red * sizeof(int32_t)));

      d_nonoverlap_size = d_snapshot_size-d_overlap_size;
      set_history(d_overlap_size+1);
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
      gr_complex *out = (gr_complex *) output_items[0];

      // Prepare the elements for the first output matrix
      std::vector<const gr_complex *> in_data_ptr(d_num_inputs);
      for (int k = 0; k < d_num_inputs; k++) {
        in_data_ptr[k] = ((gr_complex *)input_items[k]);
        prepareInData(&in_data_cnx[k * 2 * n_cols], in_data_ptr[k], n_cols, padding);
      }
      connex->writeDataToArray(in_data_cnx, total_LS_used, 0);
      executeLocalKernel(connex, autocorrelation_kernel);

      gr_complex *out_data;
      int32_t *curr_out_data_cnx;
      const int last_out_matrix = output_matrices - 1;

      // Processing for all but the last output matrix
      for (int i = 0; i < last_out_matrix; i++)
      {
        // Start thread that will do the processing for the last matrix
        std::thread t(&autocorrelateConnexKernel_impl::prepareWriteExecute,
          this, std::ref(input_items), i+1);

        out_data = &out[i * d_num_inputs * d_num_inputs];

        // Read reduction and process output data
        curr_out_data_cnx = out_data_cnx;
        for (int cnt_row = 0; cnt_row < n_rows; cnt_row++) {
          for (int cnt_col = cnt_row; cnt_col < n_rows; cnt_col++) {
            connex->readMultiReduction(n_red_per_elem, curr_out_data_cnx);

            int curr_idx = cnt_row + cnt_col * n_rows;
            out_data[curr_idx] = prepareAndProcessOutData(curr_out_data_cnx, n_red_per_elem / 2);

            // Hermitian matrix => a_ij = conj(a_ji);
            out_data[cnt_col + cnt_row * n_rows] =
              gr_complex(out_data[curr_idx].real(), -out_data[curr_idx].imag());

            curr_out_data_cnx += n_red_per_elem;
          }
        }

        // Tell the thread we have finished reading the reductions
        {
          std::lock_guard<std::mutex> lk(m);
          red_finish = true;
        }
        cv.notify_one();

        // Averaging results
        if (d_avg_method) {
          averageResults(out_data);
        } // end loop for each output matrix

        t.join();
      }

      // Processing for the last matrix
      out_data = &out[last_out_matrix * d_num_inputs * d_num_inputs];

      // Read reduction and process output data
      curr_out_data_cnx = out_data_cnx;
      for (int cnt_row = 0; cnt_row < n_rows; cnt_row++) {
        for (int cnt_col = cnt_row; cnt_col < n_rows; cnt_col++) {
          connex->readMultiReduction(n_red_per_elem, curr_out_data_cnx);

          int curr_idx = cnt_row + cnt_col * n_rows;
          out_data[curr_idx] = prepareAndProcessOutData(curr_out_data_cnx, n_red_per_elem / 2);

          // Hermitian matrix => a_ij = conj(a_ji);
          out_data[cnt_col + cnt_row * n_rows] =
            gr_complex(out_data[curr_idx].real(), -out_data[curr_idx].imag());

          curr_out_data_cnx += n_red_per_elem;
        }
      }

      // Averaging results
      if (d_avg_method) {
        averageResults(out_data);
      } // end loop for each output matrix

      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (d_nonoverlap_size*output_matrices);

      // Tell runtime system how many output items we produced.
      return (output_matrices);
    }

    void autocorrelateConnexKernel_impl::prepareWriteExecute(
      gr_vector_const_void_star &input_items,
      int nr_in_item)
    {
      uint16_t *data_cnx = static_cast<uint16_t *>(malloc(n_elems_c * sizeof(uint16_t)));
      std::vector<const gr_complex *> in_data_ptr(d_num_inputs);

      for (int k = 0; k < d_num_inputs; k++) {
        in_data_ptr[k] = ((gr_complex *)input_items[k] + nr_in_item * d_nonoverlap_size);
        prepareInData(&data_cnx[k * 2 * n_cols], in_data_ptr[k], n_cols, padding);
      }

      // Sync barrier here -- The writing and the launch must be executed only
      // after reading the reduction in the main thread
      std::unique_lock<std::mutex> lk(m);
      cv.wait(lk, []{return red_finish;});

      connex->writeDataToArray(data_cnx, total_LS_used, 0);
      executeLocalKernel(connex, "autocorrelationKernel");

      free(data_cnx);
      lk.unlock();
    }

    void autocorrelateConnexKernel_impl::prepareInData(
      uint16_t *out_data, const gr_complex *in_data, const int &n_elems_in,
      const int &padding_)
    {
      int n_elems_out_util = 2 * n_elems_in;
      for (int i = 0; i < n_elems_in; i++) {
        out_data[2 * i] = static_cast<uint16_t>(in_data[i].real() * factor_mult);
        out_data[2 * i + 1] = static_cast<uint16_t>(in_data[i].imag() * factor_mult);
      }
      if (padding != 0)
        for (int i = n_elems_out_util; i < n_elems_out_util + padding_; i++) {
          out_data[i] = 0; // padding with zeros
        }
    }

    void autocorrelateConnexKernel_impl::prepareOutData(
      gr_complex *out_data, const int32_t *in_data, const int &n_elems_in)
    {
      float temp_real, temp_imag;
      for (int i = 0; i < n_elems_in; i+=2) {
        temp_real = static_cast<float>(in_data[i]) / factor_res;
        temp_imag = static_cast<float>(in_data[i + 1]) / factor_res;

        out_data[i / 2] = gr_complex(temp_real, temp_imag);
      }
    }

    void autocorrelateConnexKernel_impl::printOutData(
      const uint16_t *in_data, const int &n_elems_in)
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
      const int32_t *in_data, const int &n_elems_in)
    {
      float temp_real, temp_imag;
      float acc_real = 0, acc_imag = 0;
      for (int i = 0; i < n_elems_in; i+=2) {
        temp_real = static_cast<float>(in_data[i]) / factor_res;
        temp_imag = static_cast<float>(in_data[i + 1]) / factor_res;

        acc_real += temp_real;
        acc_imag += temp_imag;
      }

      // Divide by the snapshot size = n_cols
      acc_real = acc_real / n_cols;
      acc_imag = acc_imag / n_cols;

      return gr_complex(acc_real, acc_imag);
    }

    void autocorrelateConnexKernel_impl::averageResults(gr_complex *out_data)
    {
      std::complex<float> two_c = (2.0, 2.0);
      std::vector<std::vector<gr_complex>> refl_matrix;
      refl_matrix.resize(n_rows);

      for (int cnt_row = 0; cnt_row < n_rows; cnt_row++) {
        refl_matrix[cnt_row].resize(n_rows);

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
    }

    int autocorrelateConnexKernel_impl::executeLocalKernel(
      ConnexMachine *connex, std::string kernel_name)
    {
      try {
        connex->executeKernel(kernel_name.c_str());
      } catch(std::string e) {
        std::cout << e << std::endl;
        return -1;
      }
      return 0;
    }

    void autocorrelateConnexKernel_impl::initKernel(const int &LS_per_row_) {
      BEGIN_KERNEL("initKernel");
        EXECUTE_IN_ALL(
          // Some constants
          R0 = LS_per_row_;
          R29 = 1;
          R28 = 0;
          R25 = 2;
        )
      END_KERNEL("initKernel");
    }

    void autocorrelateConnexKernel_impl::autocorrelationKernel(
      const int &iterations_per_array)
    {
      BEGIN_KERNEL("autocorrelationKernel");
        for (int i = 0; i < n_rows; i++) {
          for (int j = i; j < n_rows; j++) {
            EXECUTE_IN_ALL(
              R14 = i;
              R15 = j;

              R30 = R14 * R0;           // go to line i
              R31 = R15 * R0;           // go to col j
            )

            REPEAT_X_TIMES(iterations_per_array);
              EXECUTE_IN_ALL(
                R1 = LS[R30];
                R2 = LS[R31];
                R3 = R1 * R2;
                R3 = MULT_HIGH();       // re1 * re2, im1 * im2

                CELL_SHL(R2, R29);
                NOP;
                R4 = SHIFT_REG;
                R4 = R4 * R1;
                R4 = MULT_HIGH();       // re1 * im2
                R4 = R28 - R4;          // The column is conjugated => negate these

                CELL_SHR(R2, R29);
                NOP;
                R5 = SHIFT_REG;
                R5 = R5 * R1;
                R5 = MULT_HIGH();       // re2 * im1;

                R6 = INDEX;             // Select only the odd PEs
                R6 = R6 & R29;
                R7 = (R6 == R29);
                NOP;
              )

              EXECUTE_WHERE_EQ(
                R4 = R5;                // All partial imaginary parts are now in R4
              )

              EXECUTE_IN_ALL(
                REDUCE(R3);
                REDUCE(R4);

                R30 = R30 + R29;        // Go to next chunk on line & col
                R31 = R31 + R29;
              )
            END_REPEAT;
          }
        }
      END_KERNEL("autocorrelationKernel");
    }
  } /* namespace doa */
} /* namespace gr */

