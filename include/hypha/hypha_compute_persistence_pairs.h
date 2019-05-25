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
#include <phat/persistence_pairs.h>

namespace hypha {
    template<typename ReductionAlgorithm, typename Representation>
    double
    hypha_compute_persistence_pairs(ReductionAlgorithm hypha_reduction, phat::persistence_pairs &pairs, hypha::hypha_boundary_matrix <Representation> &boundary_matrix) {

        Stopwatch sw(true);
        hypha_reduction.compute(boundary_matrix);
        sw.stop();
        pairs.clear();
        for (index idx = 0; idx < boundary_matrix.get_num_cols(); idx++) {
            if (!boundary_matrix.is_empty(idx)) {
                index birth = boundary_matrix.get_max_index(idx);
                index death = idx;
                pairs.append_pair(birth, death);
            }
        }
        return sw.ms() / 1000.0;
    }

    void dualize_persistence_pairs(phat::persistence_pairs &pairs, const index n) {
        for (index i = 0; i < pairs.get_num_pairs(); ++i) {
            std::pair <index, index> pair = pairs.get_pair(i);
            pairs.set_pair(i, n - 1 - pair.second, n - 1 - pair.first);
        }
    }
}
