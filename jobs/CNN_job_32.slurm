#!/bin/bash -l
#SBATCH -J CNN_32
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 744 &
python  ${path}/main.py 745 &
python  ${path}/main.py 746 &
python  ${path}/main.py 747 &
python  ${path}/main.py 748 &
python  ${path}/main.py 749 &
python  ${path}/main.py 750 &
python  ${path}/main.py 751 &
python  ${path}/main.py 752 &
python  ${path}/main.py 753 &
python  ${path}/main.py 754 &
python  ${path}/main.py 755 &
python  ${path}/main.py 756 &
python  ${path}/main.py 757 &
python  ${path}/main.py 758 &
python  ${path}/main.py 759 &
python  ${path}/main.py 760 &
python  ${path}/main.py 761 &
python  ${path}/main.py 762 &
python  ${path}/main.py 763 &
python  ${path}/main.py 764 &
python  ${path}/main.py 765 &
python  ${path}/main.py 766 &
python  ${path}/main.py 767 &
wait
