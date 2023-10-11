# Vanilla optimizer (SGD with Nesterov Accerlation)
python main.py --seed 1 --data_name CIFAR-100 --num_classes 100 --arch resnet18 --optimizer SGD --learning_rate 0.05 --batch_size_train 128 --weight_decay 5.0e-4 --gpu_id 0 --epochs 200 --project test --scheduler cosine --use_cutmix True

# Lookahead
python main.py --seed 1 --data_name CIFAR-100 --num_classes 100 --arch resnet18 --optimizer SGD --learning_rate 0.05 --batch_size_train 128 --weight_decay 5.0e-4 --gpu_id 0 --epochs 200 --project test --scheduler cosine --use_la 1 --k 5 --alpha 0.8 --use_cutmix True

# Sharpness-aware Minimization (SAM)
python main.py --seed 1 --data_name CIFAR-100 --num_classes 100 --arch resnet18 --optimizer SGD --learning_rate 0.05 --batch_size_train 128 --weight_decay 5.0e-4 --gpu_id 0 --epochs 200 --project test --scheduler cosine --use_sam 1 --sam_rho 0.05 --use_cutmix True

# Our proposed optimizer (SALA)
python main.py --seed 1 --data_name CIFAR-100 --num_classes 100 --arch resnet18 --optimizer SGD --learning_rate 0.05 --batch_size_train 128 --weight_decay 5.0e-4 --gpu_id 0 --epochs 200 --project test --scheduler cosine --use_sala 1 --k 2 --alpha 0.5 --sala_rho 0.2  --use_cutmix True
