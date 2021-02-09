#!/bin/bash
filename=$1
ref=$2
samtools sort $filename.bam -o $filename.sorted.bam
bedtools genomecov -ibam $filename.sorted.bam -bg > $filename.bedGraph
LC_COLLATE=C sort -k1,1 -k2,2n $filename.bedGraph > $filename.sorted.bedGraph
bedGraphToBigWig $filename.sorted.bedGraph $2 $filename.bw

rm $filename.sorted.bam
rm $filename.bedGraph
rm $filename.sorted.bedGraph
