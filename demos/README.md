# 実機デモデータ

このディレクトリには実機 F1Tenth から転送したデモデータを置きます。

## ファイル構成

| ファイル | 説明 |
|---|---|
| `demo_YYYYMMDD_HHMMSS.npz` | 実機録画生データ (scans/steers/speeds/stamps) |
| `converted_demo_*.npz` | シム互換変換済みデータ (observations/actions) |

## 使い方

```bash
# 1. 実機で録画 (実機上で実行)
python3 record_demo.py

# 2. PC へ転送
scp f1tenth@CAR_IP:~/demos/demo_*.npz demos/

# 3. シム互換に変換 (Docker内)
docker compose exec f1-sim-2004 python3 scripts/convert_demo.py --all

# 4. BC プレトレーニング
docker compose exec f1-sim-2004 python3 scripts/pretrain_bc.py --algo td3

# 5. RL ファインチューニング
docker compose exec f1-sim-2004 python3 scripts/train.py --algo td3 --resume bc_pretrained
```

> ⚠️ `.npz` ファイルは `.gitignore` で除外されています。Git LFS か手動管理で共有してください。
