#!/bin/bash -l
#SBATCH -J RNN_36
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1260 &
python  ${path}/main.py 1261 &
python  ${path}/main.py 1262 &
python  ${path}/main.py 1263 &
python  ${path}/main.py 1264 &
python  ${path}/main.py 1265 &
python  ${path}/main.py 1266 &
python  ${path}/main.py 1267 &
python  ${path}/main.py 1268 &
python  ${path}/main.py 1269 &
python  ${path}/main.py 1270 &
python  ${path}/main.py 1271 &
python  ${path}/main.py 1272 &
python  ${path}/main.py 1273 &
python  ${path}/main.py 1274 &
python  ${path}/main.py 1275 &
python  ${path}/main.py 1276 &
python  ${path}/main.py 1277 &
python  ${path}/main.py 1278 &
python  ${path}/main.py 1279 &
python  ${path}/main.py 1280 &
python  ${path}/main.py 1281 &
python  ${path}/main.py 1282 &
python  ${path}/main.py 1283 &
python  ${path}/main.py 1284 &
python  ${path}/main.py 1285 &
python  ${path}/main.py 1286 &
python  ${path}/main.py 1287 &
python  ${path}/main.py 1288 &
python  ${path}/main.py 1289 &
python  ${path}/main.py 1290 &
python  ${path}/main.py 1291 &
python  ${path}/main.py 1292 &
python  ${path}/main.py 1293 &
python  ${path}/main.py 1294 &
python  ${path}/main.py 1295 &
wait
