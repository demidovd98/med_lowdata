# Running

For training and evaluation, please, follow the below-mentioned respective instructions.

NOTE 1: In case you have multiple CUDA versions installed, please, make sure to initialise the appropriate system CUDA version before running any command.
```bash
# <10.2> - CUDA version number
module load cuda-10.2
```

NOTE 2: Make sure that you are in the root repo's directory '.../med_lowdata/':
```
cd <your_path>/med_lowdata/
```

NOTE 3: Make sure that the 'med_lowdata' conda environment is activated (see [INSTALL.md](INSTALL.md) ):
```
conda activate med_lowdata
```

<hr />


## CRC

### Train + Test (Vanilla model):

```bash
python3 -W ignore -m torch.distributed.launch --nproc_per_node 1 train.py --name CRC_test --dataset CRC --img_size 196 --resize_size 224 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.01 --num_steps 5000 --fp16 --eval_every 200 --vanilla --split '<split>' --data_root '<dataset_path>/CRC_colorectal_cancer_histology'
```
Where <split> is '<percentage>_sp<number_of_drawn_split>'. Example: '3_sp1'

### Train + Test (Our model):
```bash
python3 -W ignore -m torch.distributed.launch --nproc_per_node 1 train.py --name CRC_test --dataset CRC --img_size 196 --resize_size 224 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.01 --num_steps 5000 --fp16 --eval_every 200 --split '<split>' --data_root '<dataset_path>/CRC_colorectal_cancer_histology'
```
Where <split> is '<percentage>_sp<number_of_drawn_split>'. Example: '3_sp1'

### Test only:
```bash
soon
```


<hr />

