"""
CPUスレッド数の確認スクリプト
"""
import sys
import os
import multiprocessing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config

print(f"コア数        : {multiprocessing.cpu_count()}")
print(f"使用スレッド数: {config.TORCH_NUM_THREADS}")
print()
if multiprocessing.cpu_count() > 12:
    print("→ 高コア数マシン検出: スレッド数を制限して最適化済み (Desktop)")
else:
    print("→ 標準コア数マシン: フルスレッドを使用 (Laptop)")
print(f"  変更する場合: TORCH_NUM_THREADS=<数> python3 scripts/train.py")