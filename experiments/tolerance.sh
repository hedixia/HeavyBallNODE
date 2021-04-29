tolerances=(
    0.001
    0.01
    0.1
)

for tol in  ${tolerances[@]}; do
	echo Running $1 tolerance $tol . . . on device $2
	python ../source/main.py --model $1 --gpu $2 --tol $tol
done
