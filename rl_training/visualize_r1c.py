import numpy as np
import matplotlib.pyplot as plt

def calculate_rc1(distance, k_4, rho_0, r_safe, eta_r):
    """
    ユーザー提供の数式を計算する関数
    distance: 障害物までの距離 (配列または単一数値)
    """
    # 数式内のゼロ除算を防ぐため、安全マージン以下の距離での挙動に注意が必要ですが、
    # ここでは数式の挙動そのものを可視化します。
    
    # 計算項 (1/d - 1/rho_0)
    term_dist = (1 / distance) - (1 / rho_0)
    
    # 正規化項 (1/r_safe - 1/rho_0)
    term_norm = (1 / r_safe) - (1 / rho_0)
    
    # 全体の計算
    # numpyの配列操作に対応させるため、そのまま計算します
    r_c_1 = -k_4 * (term_dist / term_norm) ** eta_r
    
    return r_c_1

# --- パラメータ設定 (これらは想定値です。実際のシステムに合わせて変更してください) ---
k_4 = 1.0       # ゲイン (強さの係数)
rho_0 = 5.0     # 影響範囲 (この距離より遠いと通常は影響なしとする)
r_safe = 0.5    # 安全距離 (これより近づくと値が急変する基準)
eta_r = 2.0     # 指数 (カーブの急峻さを決める)

# --- データの生成 ---
# r_safe から rho_0 までの範囲をプロットします
# r_safe に極端に近づくと 1/d が大きくなりすぎるため、r_safeの少し上から開始します
distances = np.linspace(r_safe + 0.01, rho_0, 500)

# 計算実行
values = calculate_rc1(distances, k_4, rho_0, r_safe, eta_r)

# --- プロット ---
plt.figure(figsize=(10, 6))
plt.plot(distances, values, label=f'$\eta_r={eta_r}, k_4={k_4}, \\rho_0={rho_0}, r_{{safe}}={r_safe}$')

# グラフの装飾
plt.xlabel('Distance to Obstacle ($d^{curr}_{to\_obstacle}$)')
plt.ylabel('$r^c_1$')
plt.axvline(x=rho_0, color='green', linestyle='--', label=r'$\rho_0$ (Influence Limit)')
plt.axvline(x=r_safe, color='red', linestyle='--', label=r'$r_{safe}$ (Safety Margin)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

# 表示
plt.show()