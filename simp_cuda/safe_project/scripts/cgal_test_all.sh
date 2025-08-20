# for all .obj paths in the input directory, run the cgal script

input_dir=$1
output_path=$2

# cgal script
cgal_script=./scripts/cgal_intersect/cgal_intersect

# for all .obj files in the input directory
for obj in $input_dir/*.obj; do
    # get the filename
    filename=$(basename $obj)
    # run the cgal script
    $cgal_script $obj $output_path
done