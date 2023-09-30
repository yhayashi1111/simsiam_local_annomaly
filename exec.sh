python main_simsiam_local.py -a resnet50 --dist-url 'tcp://localhost:10001' --lamda 0.1 --save_dir spd_0.1/ --multiprocessing-distributed --world-size 1 --rank 0 --fix-pred-lr /data/imagenet



