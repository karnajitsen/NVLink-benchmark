#!/bin/bash
#BSUB -J nvlink # job name
#BSUB -q S824L-K80
#BSUB -cwd /gpfs/home/ksen/karna/mThesis/tool
#BSUB -oo openmpi.stdout.%J
#BSUB -eo mpich.stderr.%J

make ref0

