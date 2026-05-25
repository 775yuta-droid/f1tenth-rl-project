import numpy as np
from ..racing_line import RacingLine

class PurePursuitController:
    """
    レーシングラインを追従するための Pure Pursuit 制御器。
    """
    def __init__(self, racing_line: RacingLine, wheelbase: float = 0.33, lookahead_dist: float = 0.8):
        """
        Args:
            racing_line: RacingLine オブジェクト
            wheelbase: 車両のホイールベース (m)
            lookahead_dist: 先読み距離 (m)
        """
        self.racing_line = racing_line
        self.wheelbase = wheelbase
        self.lookahead_dist = lookahead_dist

    def get_base_action(self, x: float, y: float, yaw: float, current_speed: float, max_speed: float, min_speed: float):
        """
        現在の車両状態からベースとなるアクションを計算する。
        
        Returns:
            steer_rad: ステアリング角 (rad)
            speed: 目標速度 (m/s)
        """
        if not self.racing_line._loaded:
            return 0.0, min_speed

        # --- 先読み点 (Lookahead Point) の検索 ---
        # 1. 最近傍点を見つける
        idx = self.racing_line._find_nearest(x, y)
        N = len(self.racing_line.xy)
        
        # 2. 現在地から lookahead_dist 先にあるライン上の点を探す
        target_idx = idx
        for i in range(1, N):
            check_idx = (idx + i) % N
            dist = np.sqrt((self.racing_line.xy[check_idx, 0] - x)**2 + 
                           (self.racing_line.xy[check_idx, 1] - y)**2)
            if dist >= self.lookahead_dist:
                target_idx = check_idx
                break
        
        target_wp = self.racing_line.xy[target_idx]

        # --- ステアリング角の計算 ---
        # 目標点を車両座標系に変換
        dx = target_wp[0] - x
        dy = target_wp[1] - y
        
        # 車両前方方向を X 軸とする座標系への回転
        local_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
        local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)
        
        # L2 距離
        L = np.sqrt(local_x**2 + local_y**2)
        
        # 曲率 kappa = 2 * dy / L^2
        if L > 0:
            kappa = 2.0 * local_y / (L**2)
        else:
            kappa = 0.0
            
        # ステアリング角 delta = atan(kappa * L) ではなく、物理的なホイールベースに基づく
        # delta = atan(kappa * wheelbase)
        steer_rad = np.arctan(kappa * self.wheelbase)

        # --- 速度の計算 (曲率ベース) ---
        # ターゲット点の曲率を取得
        target_curvature = abs(self.racing_line.curvature[target_idx])
        
        # シンプルな速度プロファイル: v = v_max * exp(-k * curvature)
        # 曲率が高いほど減速する
        speed = max_speed * np.exp(-1.8 * target_curvature)
        speed = np.clip(speed, min_speed, max_speed)
        
        return steer_rad, speed
