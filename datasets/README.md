The script bam_to_bw.sh takes a bam file and creates a bw file which is later refered to as raw read bw (in contrast to fold and sign which are ENCODE processed bw files)
To run the script the following installation lines might be useful:

`#!/bin/bash
URL=http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64

curl $URL/bedGraphToBigWig > ~/bin/bedGraphToBigWig

chmod +x ~/bin/bedGraphToBigWig

curl $URL/faToTwoBit > ~/bin/faToTwoBit

chmod +x ~/bin/faToTwoBit`

