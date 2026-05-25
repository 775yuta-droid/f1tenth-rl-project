#dockerの起動
docker compose up -d

#docker にはいる
docker compose exec f1-sim-2004 bash

#学習開始
python3 scripts/train.py --model A --resume B --steps 10000000

#可視化
python3 scripts/enjoy_wide.py --model A --steps 500 --save A.mp4

#監視（コンテナ内で）
tensorboard --logdir logs --host localhost

