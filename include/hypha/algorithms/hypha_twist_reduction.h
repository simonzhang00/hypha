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
    class hypha_twist_reduction{
    public:

        template< typename Representation >
        void compute(hypha::hypha_boundary_matrix< Representation >& boundary_matrix) {

            printf("RUNNING HYPHA-TWIST\n");
            const index nr_columns = boundary_matrix.get_num_cols();

            Stopwatch sw(true);
            boundary_matrix.GPU_scan(nr_columns);//find stable, unstable columns and destroyer indices
            sw.stop();
            std::cout << sw.ms() << "ms for GPU scan\n";


            //the following parallel for loop results in double free for the following data structures:
            //vector_heap, vector_list, vector_set
#pragma omp parallel for schedule( guided, 1 )
            for (index i = 0; i < nr_columns; i++) {
                //apply twist on CPU side
                index positive_row = boundary_matrix.get_max_index(i);
                if (positive_row >= 0) {
                    boundary_matrix.clear(positive_row);
                    boundary_matrix.set_dim(positive_row, 0);
                }
            }
            if(boundary_matrix.with_compression()){
                //search for compressible entries
                std::vector <index_type> is_compressible(nr_columns, UNKNOWN);

                _find_compressible(boundary_matrix, boundary_matrix.unstables, is_compressible);

                std::vector <index> tmp_col;
#pragma omp parallel for schedule( guided, 1 ), private(tmp_col )
                for (index j = 0; j < *boundary_matrix.unstable_nr; j++) {
                    index i = boundary_matrix.unstables[j];
                    _compression(i, boundary_matrix, is_compressible, tmp_col);
                }
                boundary_matrix.sync();
            }

            for (index cur_dim = boundary_matrix.get_max_dim(); cur_dim >= 1; cur_dim--) {
                for (index i = 0; i < *boundary_matrix.unstable_nr; i++) {
                    index cur_col = boundary_matrix.unstables[i];
                    //recall: stables[i]= 0 if unstable, 1 if stable, and -1 if 0 column
                    if (boundary_matrix.get_dim(cur_col) == cur_dim) {
                        index lowest_one = boundary_matrix.get_max_index(cur_col);
                        while (lowest_one != -1 && boundary_matrix.lookup[lowest_one] > -1) {
                            boundary_matrix.add_to(boundary_matrix.lookup[lowest_one], cur_col);
                            lowest_one = boundary_matrix.get_max_index(cur_col);
                        }
                        if (lowest_one != -1) {
                            boundary_matrix.lookup[lowest_one] = cur_col;
                            boundary_matrix.clear(lowest_one);
                        }
                        boundary_matrix.finalize(cur_col);
                    }
                }
            }
        }
    };
}
