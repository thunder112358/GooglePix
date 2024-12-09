Compile Successfully;
Segmentation Fault;
Developing


Usage: 
gcc -o verify_real_image verify_real_image.c block_matching.c ica.c -lm

./verify_real_image left.yuv right.yuv 1920 1080



Or more fully,

make -f makefile_verify_real 
./verify_real_image left.yuv right.yuv 1920 1080

