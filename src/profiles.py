"""
学習環境プロファイル設定

環境変数 TRAINING_PROFILE で使用するプロファイルを切り替えます。
  laptop  : RTX 3050 Laptop 向け（低スレッド数でFPS最適化）
  desktop : RTX 5080 Desktop 向け（高スレッド数で学習スループット向上）

例:
  TRAINING_PROFILE=laptop  python3 scripts/train.py
  TRAINING_PROFILE=desktop python3 scripts/train.py

未設定の場合は auto プロファイルが適用されます（CPUコア数から自動判定）。
"""

# プロファイル別の TORCH_NUM_THREADS 設定
PROFILES: dict[str, dict] = {
    "laptop": {
        "torch_num_threads": 2,  # RTX 3050 Laptop: スレッド競合を避けて2に固定
    },
    "desktop": {
        "torch_num_threads": 4,  # RTX 5080 Desktop: 少し多めでも余裕あり
    },
    "auto": {
        # config.py 側でCPUコア数から動的に決定するため None を指定
        "torch_num_threads": None,
    },
}
