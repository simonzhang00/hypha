/*  Copyright 2018, 2019 The Ohio State University
    Contributed by: Simon Zhang, Mengbai Xiao, Chengxin Guo, Liang Geng, Hao Wang, Xiaodong Zhang
    HYPHA makes calls to the PHAT library: (Ulrich Bauer, Michael Kerber, Jan Reininghaus, Hubert Wagner)

    This file is part of HYPHA.

    HYPHA is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    HYPHA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with HYPHA.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once

#include <hypha/hypha_boundary_matrix.h>

namespace hypha {
    class hypha_spectral_sequence_reduction {
    public:
        template< typename Representation >
        void compute(hypha::hypha_boundary_matrix< Representation >& boundary_matrix) {

            printf("RUNNING HYPHA-SS\n");
            const index nr_columns = boundary_matrix.get_num_cols();

            Stopwatch sw(true);
            boundary_matrix.GPU_scan(nr_columns);//find stable, unstable columns and destroyer indices
            sw.stop();
            std::cout << sw.ms() << "ms for GPU scan\n";

            //the following parallel for loop results in double free for the following data structures:
            //vector_heap, vector_list, vector_set
            //do clearing in parallel.
#pragma omp parallel for schedule( guided, 1 )
            for (int i = 0; i < nr_columns; i++) {
                int positive_row = boundary_matrix.get_max_index(i);
                //apply twist on CPU side
                if (positive_row >= 0) {
                    boundary_matrix.clear(positive_row);
                    boundary_matrix.set_dim(positive_row, 0);
                }
            }


            if(boundary_matrix.with_compression()) {
                //search for compressible entries
                std::vector<index_type> is_compressible(nr_columns, UNKNOWN);

                _find_compressible(boundary_matrix, boundary_matrix.unstables, is_compressible);

                std::vector<index> tmp_col;
#pragma omp parallel for schedule( guided, 1 ), private(tmp_col )
                for (int j = 0; j < *boundary_matrix.unstable_nr; j++) {
                    int i = boundary_matrix.unstables[j];
                    _compression(i, boundary_matrix, is_compressible, tmp_col);
                }
                boundary_matrix.sync();
            }


            const index num_stripes = omp_get_max_threads();

            index block_size = (nr_columns % num_stripes == 0) ? nr_columns / num_stripes : block_size = nr_columns / num_stripes + 1;

            std::vector<index> unstables_filtration(num_stripes+1);
            std::vector<index> unstables_vec(boundary_matrix.unstables, boundary_matrix.unstables+*boundary_matrix.unstable_nr);

#pragma omp parallel for schedule(guided, 1)
            for(index cur_stripe= 0; cur_stripe<num_stripes; cur_stripe++){
                //find smallest index >= cur_stripe*block_size
                std::vector<index>::iterator ublockstart= std::lower_bound(unstables_vec.begin(), unstables_vec.end(),cur_stripe * block_size);
                unstables_filtration[cur_stripe]= ublockstart-unstables_vec.begin();
            }
            unstables_filtration[num_stripes]= nr_columns;

            std::vector <std::vector<index>> unreduced_cols_cur_pass(num_stripes);
            std::vector <std::vector<index>> unreduced_cols_next_pass(num_stripes);
            for( index cur_dim = boundary_matrix.get_max_dim(); cur_dim >= 1 ; cur_dim-- ) {
#pragma omp parallel for schedule( guided, 1 )
                for( index cur_stripe = 0; cur_stripe < num_stripes; cur_stripe++ ) {
                    for(index u= unstables_filtration[cur_stripe]; u<unstables_filtration[cur_stripe+1]; u++) {
                        index cur_col= boundary_matrix.unstables[u];
                        if (boundary_matrix.get_dim(cur_col) == cur_dim && boundary_matrix.get_max_index(cur_col) != -1) {
                            unreduced_cols_cur_pass[cur_stripe].push_back(cur_col);
                        }
                    }
                }

                for( index cur_pass = 0; cur_pass < num_stripes; cur_pass++ ) {
                    boundary_matrix.sync();
#pragma omp parallel for schedule( guided, 1 )
                    for (int cur_stripe = 0; cur_stripe < num_stripes; cur_stripe++) {
                        index row_begin = (cur_stripe - cur_pass) * block_size;
                        index row_end = row_begin + block_size;
                        unreduced_cols_next_pass[cur_stripe].clear();
                        for (index idx = 0; idx < (index) unreduced_cols_cur_pass[cur_stripe].size(); idx++) {
                            index cur_col = unreduced_cols_cur_pass[cur_stripe][idx];
                            index lowest_one = boundary_matrix.get_max_index(cur_col);
                            while (lowest_one != -1 && lowest_one >= row_begin &&
                                   lowest_one < row_end && boundary_matrix.lookup[lowest_one] > -1) {
                                boundary_matrix.add_to(boundary_matrix.lookup[lowest_one], cur_col);
                                lowest_one = boundary_matrix.get_max_index(cur_col);
                            }
                            if (lowest_one != -1) {
                                if (lowest_one >= row_begin && lowest_one < row_end) {
                                    boundary_matrix.lookup[lowest_one] = cur_col;
                                    boundary_matrix.clear( lowest_one );
                                    boundary_matrix.finalize(cur_col);
                                } else {
                                    unreduced_cols_next_pass[cur_stripe].push_back(cur_col);
                                }
                            }
                        }
                        unreduced_cols_next_pass[cur_stripe].swap(unreduced_cols_cur_pass[cur_stripe]);
                    }
                }
            }
            //ADD THIS TO UPDATE BOUNDARY_MATRIX ON TIME (bug in the original PHAT code)
            boundary_matrix.sync();

        }
    };
}
