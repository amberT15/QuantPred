The script bam_to_bw.sh takes a bam file and creates a bw file which is later refered to as raw read bw (in contrast to fold and sign which are ENCODE processed bw files)
To run the script the following installation lines might be useful:

```
#!/bin/bash
URL=http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64
curl $URL/bedGraphToBigWig > ~/bin/bedGraphToBigWig
chmod +x ~/bin/bedGraphToBigWig
curl $URL/faToTwoBit > ~/bin/faToTwoBit
chmod +x ~/bin/faToTwoBit
```



The script bw_to_tfr.sh and the wrapper script create_25TF_datasets.py run out of memory unless the following is ran 

```
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1
```
