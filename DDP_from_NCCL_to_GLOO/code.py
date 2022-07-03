array=( resnet50 )
epoch=10
for i in "${array[@]}"
do
    echo $i
    for j in 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=20480 python -u train.py -n 1 -g 2 -nr 0 -e ${epoch} --batch-size 32 -m $i --backend nccl &> ./${i}_nccl_2gpu_${j}.txt
        CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=20481 python -u train.py -n 1 -g 2 -nr 0 -e ${epoch} --batch-size 32 -m $i --backend gloo &> ./${i}_gloo_2gpu_${j}.txt
    done
done
