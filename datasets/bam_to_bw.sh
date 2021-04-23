#!/bin/bash

folder_path=$1
ref=$2
output_dir=$3

for FILE in $folder_path*.bam; do
  [ -f "$FILE" ] || break

  # echo Processing $FILE
  filepath="${FILE%.*}"
  exp_id="${filepath##*/}"
  output_path="$output_dir/$exp_id.bw"
  echo Sorting bam file $FILE
  samtools sort $filepath.bam -o $filepath.sorted.bam
  echo Converting bam to bedGraph
  bedtools genomecov -ibam $filepath.sorted.bam -bg > $filepath.bedGraph
  LC_COLLATE=C sort -k1,1 -k2,2n $filepath.bedGraph > $filepath.sorted.bedGraph
  echo Converting bedGraph to bigwig
  bedGraphToBigWig $filepath.sorted.bedGraph $ref $output_path

  rm $filepath.sorted.bam
  rm $filepath.bedGraph
  rm $filepath.sorted.bedGraph

done
