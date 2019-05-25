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

#include <phat/representations/vector_vector.h>
#include <phat/boundary_matrix.h>
#include <phat/helpers/dualize.h>
#include <hypha/helpers/stopwatch.h>

void print_help() {
    std::cerr << "Usage: " << "convert " << "[options] input_filename output_filename" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << std::endl;
    std::cerr << "--ascii   --  use ascii file format for input_filename" << std::endl;
    std::cerr << "--binary  --  use binary file format for input_filename (default)" << std::endl;
    std::cerr << "--save-ascii   --  use ascii file format for output_filename" << std::endl;
    std::cerr << "--save-binary  --  use binary file format for output_filename (default)" << std::endl;
    std::cerr << "--dualize --  dualize filtration" << std::endl;
    std::cerr << "--help    --  prints this screen" << std::endl;
}

void print_help_and_exit() {
    print_help();
    exit( EXIT_FAILURE );
}

void parse_command_line( int argc, char** argv, bool& use_binary, bool& use_save_binary, bool& use_dualize, bool& verbose, std::string& input_filename, std::string& output_filename) {

    if( argc < 3 ) print_help_and_exit();

    input_filename = argv[ argc - 2 ];
    output_filename = argv[ argc - 1 ];

    for( int idx = 1; idx < argc - 2; idx++ ) {
        const std::string option = argv[ idx ];

        if( option == "--dualize" ) use_dualize = true;
        else if( option == "--ascii" ) use_binary = false;
        else if( option == "--binary" ) use_binary = true;
        else if( option == "--save-ascii" ) use_save_binary = false;
        else if( option == "--save-binary" ) use_save_binary = true;
        else if( option == "--help" ) print_help_and_exit();
        else if( option == "--notimer" ) verbose= false;
        else print_help_and_exit();
    }
}

int main( int argc, char** argv )
{
    bool use_binary = true; // interpret input as binary or ascii file
    bool use_save_binary = true; // write output as binary or ascii file
    bool use_dualize = false; // dualize filtration
    bool verbose= true; // verbose turned on
    std::string input_filename; // name of file that contains the boundary matrix
    std::string output_filename; // name of file that will contain the boundary matrix in the new binary format

    parse_command_line( argc, argv, use_binary, use_save_binary, use_dualize, verbose, input_filename, output_filename );
    
    phat::boundary_matrix< phat::bit_tree_pivot_column > matrix;
    Stopwatch sw(true);
    if( use_binary )
        matrix.load_binary( input_filename );
    else
        matrix.load_ascii( input_filename );
    sw.stop();

    std::cout<<"Reading input file took " << (sw.ms()/1000.0) <<"s"<<std::endl;

    sw.start();
    if( use_dualize )
        dualize( matrix );
    sw.stop();
    std::cout<<"Dualizing took " << (sw.ms()/1000.0) <<"s"<<std::endl;

    sw.start();
    if( use_save_binary )
        matrix.save_binary( output_filename );
    else
        matrix.save_ascii( output_filename );
    sw.stop();
    std::cout<<"Writing output file took " << (sw.ms()/1000.0) <<"s"<<std::endl;
}
