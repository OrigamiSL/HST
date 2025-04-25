python -u main.py --data ECL --input_len 336  --pred_len 96,192,336,720,1200 --period 24,168,336 --encoder_layer 3 --layer_stack 4 --MODWT_level 2 --patch_size 6 --ccc_num 24 --d_model 128 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 1 --train --patience 2 --decay 0.5 --save_loss

python -u main.py --data Solar --input_len 288  --pred_len 96,192,336,720,1200 --period 144,144,288 --encoder_layer 3 --layer_stack 4 --MODWT_level 2 --patch_size 6 --ccc_num 24 --d_model 128 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 1 --train --patience 2 --decay 0.5 --save_loss

python -u main.py --data Wind --input_len 336  --pred_len 96,192,336,720,1200 --period 24,168,336 --encoder_layer 3 --layer_stack 4 --MODWT_level 2 --patch_size 6 --ccc_num 28 --d_model 128 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 1 --train --patience 2 --decay 0.5 --save_loss

python -u main.py --data Hydro --input_len 336  --pred_len 48,96,168,336,720 --period 24,168,336 --encoder_layer 3 --layer_stack 4 --MODWT_level 2 --patch_size 6 --ccc_num 14 --d_model 128 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 1 --train --patience 2 --decay 0.5 --save_loss
