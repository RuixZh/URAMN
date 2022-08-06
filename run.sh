python main.py  --dataset dblp --alpha 0.1 --beta 0.2 --weight_decay 0.0001 --reg_coef 0.01
python main.py  --dataset acm --alpha 0.1 --beta 0.2 --weight_decay 0.0 --reg_coef 0.1
python main.py  --dataset yelp --alpha 0.8 --beta 0.8 --weight_decay 0.0 --reg_coef 0.01

# python main.py  --dataset dblp --alpha 0.1 --beta 0.2 --weight_decay 0.0001 --isSemi --reg_coef 0.01
# python main.py  --dataset acm --alpha 0.1 --beta 0.2 --weight_decay 0.0 --isSemi --reg_coef 0.1
# python main.py  --dataset yelp --alpha 0.8 --beta 0.8 --weight_decay 0.0 --isSemi --reg_coef 0.01