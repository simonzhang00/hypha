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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <hypha/helpers/stopwatch.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>
#include <cub/grid/grid_barrier.cuh>
#include <hypha/helpers/hypha_misc.h>

namespace hypha{
    template<typename Representation>
    class hypha_boundary_matrix;

    //DFS from unstable columns' rows to determine COMPRESSIBLE and INCOMPRESSIBLE such rows
    template< typename Representation >
    void _find_compressible( const hypha::hypha_boundary_matrix< Representation >& boundary_matrix,
                             const index* unstable_columns,
                             std::vector< index_type >& is_compressible ){

        //is_compressible[] is initialized to all UNKNOWN
        std::vector<std::tuple<index,index,char>> stack;//third index tells us second or first pass
        std::vector< index > tmp_col;
        //parallelize this for loop, private stack and private tmp_col
#pragma omp parallel for schedule( guided, 1 ), private( stack, tmp_col )
        for(int j=0; j<*boundary_matrix.unstable_nr; j++){

            //local to each thread:
            int starting_column = unstable_columns[j];
            int low= boundary_matrix.get_max_index(starting_column);

            boundary_matrix.get_col(starting_column, tmp_col);
            for (int i = tmp_col.size()-1; i >=0; i--) {
                index row= tmp_col[i];
                stack.push_back(std::make_tuple(row, -1, 0));
            }

            while(!stack.empty()) {
                index cur_row= std::get<0>(stack.back());
                index parent_row= std::get<1>(stack.back());
                char secondpass= std::get<2>(stack.back());

                int pivot_col = boundary_matrix.lookup[cur_row];

                if (is_compressible[cur_row] == COMPRESSIBLE) {
                    stack.pop_back();
                    continue;
                }
                else if (is_compressible[cur_row] == INCOMPRESSIBLE) {
                    if(parent_row!=-1) {
                        is_compressible[parent_row] = INCOMPRESSIBLE;
                    }
                    stack.pop_back();
                    continue;
                }

                if(secondpass==1){
                    //if cur_row was not touched in the first pass, it must be compressible
                    is_compressible[cur_row]= COMPRESSIBLE;

                    stack.pop_back();
                    continue;
                }

                //if the column with index cur_row has a leftmost nonzero entry then cur_row is a destroyer index and is thus compressible
                if(boundary_matrix.lookup[cur_row] ==-2){
                    is_compressible[cur_row]= COMPRESSIBLE;
                    stack.pop_back();

                    continue;
                }else if(pivot_col>=0){//there exists a pivot col to the left: check that the current column is not a pivot col as well

                    boundary_matrix.get_col(pivot_col, tmp_col);
                    std::get<2>(stack.back())= 1;
                    for(int i=tmp_col.size()-2; i>=0; i--) {//ignore the lowest one
                        stack.push_back(std::make_tuple(tmp_col[i], cur_row, 0));
                    }
                    continue;
                }else {
                    is_compressible[cur_row] = INCOMPRESSIBLE;
                    if (parent_row != -1) {
                        is_compressible[parent_row] = INCOMPRESSIBLE;
                    }
                    stack.pop_back();
                    continue;
                }
            }
        }
    }

    template< typename Representation >
    void _compression( const index col_idx,
                       hypha::hypha_boundary_matrix< Representation >& boundary_matrix,
                       const std::vector< index_type >& is_compressible,
                       std::vector< index >& temp_col ){

        temp_col.clear();
        while( !boundary_matrix.is_empty( col_idx ) ) {
            index cur_row = boundary_matrix.get_max_index(col_idx);
            index pivot_col= boundary_matrix.lookup[cur_row];
            if(is_compressible[cur_row] == COMPRESSIBLE && pivot_col < col_idx && pivot_col != -1){//this includes checking pivot_col!=-1
                boundary_matrix.remove_max(col_idx);
            }
            //the below will result in race condition for columns if we compress stable columns with unstable columns in parallel, thus we only compress unstable columns and do not touch stable columns
            else if(pivot_col>-1 && pivot_col<col_idx){//add the pivot column to col_idx to cancel this entry
                boundary_matrix.add_to(pivot_col, col_idx);
            }
            else{//transfer element from current column to temp col
                temp_col.push_back(cur_row);
                boundary_matrix.remove_max(col_idx);
            }
        }
        std::reverse( temp_col.begin(), temp_col.end() );
        boundary_matrix.set_col( col_idx, temp_col );
        boundary_matrix.finalize(col_idx);
    }

    //this method searches for stable columns along with destroyer indices and generates the unstable set of columns for later computation
    //all computation on GPU side done without column additions
    //see the HYPHA paper for details on GPU algorithm
    __global__ void scan_kernel(index ** matrix, index size, index * d_leftmostnz_inrow, index * lookup,cub::GridBarrier global_barrier,index *col_num,index * psum,index *stable, index * gpu_us_num,index *gpu_unstables){

        //notice gridDim.x is under 2^31-1; It is safe to assume there are atmost 2^31-1 columns in matrix.
        //matrix with atleast 2^31 columns IS UNTESTED
        //this limits the number of columns for our algorithm.
        index stride = (index)blockDim.x * (index)gridDim.x;
        index offset = (index)blockIdx.x * (index)blockDim.x + (index)threadIdx.x;

        index col_size, pos;
        index mid_low;

        for (index col = offset; col < size; col += stride) {
            col_size = col_num[col];
            pos = psum[col];
            if (col_size == 0) {
                stable[col] = -1;// all zero -1,0 for unstable ,1 for stable
                continue;
            } else {

                for (index row = 0; row < col_size; row++) {
                    mid_low = matrix[BLOCK_ID(pos + row)][BLOCK_OFF(pos + row)];//mid_low is the row index in the uncompressed matrix

                    //the below assumes hypha/helpers/hypha_misc.h and phat/helpers/misc.h have typedef int index:
                    //cast to unsigned int to get MAXINT= 2^32-1 as infinity
                    //this must be unsigned int since we are comparing with -1= infinity
                    //uncomment the below for typedef int index;
                    ///atomicMin((unsigned int *) &d_leftmostnz_inrow[mid_low], (unsigned int) col);

                    //the below assumes that hypha/helpers/hypha_misc.h and phat/helpers/misc.h use typedef int64_t index
                    //casting to unsigned long long for comparison of 64 bit integers as indices
                    //turns -1 into infinity
                    atomicMin((unsigned long long int*)&d_leftmostnz_inrow[mid_low], (unsigned long long int)col);

                }
            }
        }
        global_barrier.Sync();

        //compute the lowest fixed entries (d_maxleftfixed)
        for (index col = offset; col < size; col += stride) {
            col_size = col_num[col];
            pos = psum[col];
            if(col_size==0)continue;
            index lowoffset = pos + col_size - 1;

            index lowforcol = matrix[BLOCK_ID(lowoffset)][BLOCK_OFF(lowoffset)];
            stable[lowforcol]= -1;
            char foundneg= 0;

            //we don't need a barrier between finding destroyer indices and finding seed pivots/stable columns
            if(col == d_leftmostnz_inrow[lowforcol]) {
                stable[col] = 1;
                lookup[lowforcol] = col;

                lookup[col]= -2;
                foundneg= 1;
            }
            if(!foundneg) {
                for (index row = col_size - 2; row >= 0; row--) {
                    mid_low = matrix[BLOCK_ID(pos + row)][BLOCK_OFF(pos + row)];

                    if (d_leftmostnz_inrow[mid_low] == col) {
                        lookup[col] = -2;
                        break;
                    }
                }
            }
        }
        global_barrier.Sync();

        for (index i = offset; i < size; i += stride) {
            if (stable[i] == 0) {
                pos = atomicAdd((unsigned long long int *)gpu_us_num, 1);
                gpu_unstables[pos] = i;
            }
        }
    }
}
