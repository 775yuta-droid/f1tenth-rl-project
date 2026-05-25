#!/usr/bin/env python3
"""
test_joystick.py — ジョイスティック軸番号確認ツール
====================================================
Futaba T4PM (またはその他 USB ジョイスティック) の軸番号とボタン番号を表示する。
F1Tenth Gym / Docker 環境不要のスタンドアロンスクリプト。

使い方:
  python3 scripts/tests/test_joystick.py

ジョイスティックを動かすと、リアルタイムで各軸の値が表示される。
ステアリング軸とスロットル軸の番号を確認したら、record_demo.py に設定する。
"""

import sys

try:
    import pygame
except ImportError:
    print("[ERROR] pygame がインストールされていません。")
    print("  pip install pygame")
    sys.exit(1)


def main():
    pygame.init()
    pygame.joystick.init()

    n_joy = pygame.joystick.get_count()
    if n_joy == 0:
        print("[ERROR] ジョイスティックが検出されません。")
        print("  USB ケーブルを接続してから再試行してください。")
        sys.exit(1)

    print(f"検出されたジョイスティック: {n_joy} 個")
    for i in range(n_joy):
        joy = pygame.joystick.Joystick(i)
        joy.init()
        print(f"  [{i}] {joy.get_name()}  "
              f"(軸: {joy.get_numaxes()}, ボタン: {joy.get_numbuttons()})")

    # デフォルトで index 0 を使用
    joy_idx = 0
    if n_joy > 1:
        try:
            joy_idx = int(input(f"使用するジョイスティック番号 [0-{n_joy-1}]: "))
        except (ValueError, EOFError):
            joy_idx = 0

    joystick = pygame.joystick.Joystick(joy_idx)
    joystick.init()
    print(f"\n使用: [{joy_idx}] {joystick.get_name()}")
    print(f"軸数: {joystick.get_numaxes()}")
    print("\n--- ジョイスティックを動かしてください ---")
    print("(Ctrl+C で終了)\n")

    header = "  ".join([f"Axis{i:02d}" for i in range(joystick.get_numaxes())])
    print(header)
    print("-" * len(header))

    try:
        while True:
            pygame.event.pump()
            vals = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
            line = "  ".join([f"{v:+.3f}" for v in vals])
            print(f"\r{line}", end="", flush=True)
            pygame.time.wait(50)

    except KeyboardInterrupt:
        print("\n\n終了しました。")
        print("\n【設定メモ】")
        print("  ステアリング軸番号: Axis__ (左右に動かしたとき変化した軸)")
        print("  スロットル軸番号  : Axis__ (前後に動かしたとき変化した軸)")
        print("  → record_demo.py の STEER_AXIS / THROTTLE_AXIS に設定してください")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
