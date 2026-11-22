# archive

開発途中の旧バージョンを保管しているフォルダです。現在のコードからは参照されていません。

- フォルダ構成は元の場所をそのまま再現しています（例: `archive/multi_robot/renderer/simulator_v4.py` は元 `multi_robot/renderer/simulator_v4.py`）。
- 相対 import が切れているため、このままでは実行できません。動かす場合は元の場所に戻してください。

| 元の場所 | 内容 | 後継 |
|---|---|---|
| `multi_robot/renderer/simulator*.py` (v1〜v4) | 旧シミュレータ | `simulator_v5*.py` |
| `multi_robot/renderer/formation_sidebyside_{1,3,6,7}.py`, `formation_previous_research.py` | 旧隊列計画 | `formation_sidebyside_5.py`（提案手法）/ `_2.py`（比較手法） |
| `multi_robot/renderer/policy.py`, `policy2.py` | 旧方策コントローラ | `policy2_newobs.py` |
| `multi_robot/models/` | 旧学習済みモデル | `multi_robot/models/model_ep16500.zip` |
| `rl_training/gym_environment{,_v2,_v3}.py`, `train_v{1,2,3}.py`, `callbacks.py`, `periodic_test.py` | 旧学習環境・学習スクリプト | `gym_environment_v4.py`, `train_v4.py`, `callbacks_v2.py` |
| `rl_training/whill_env/` | 初期の ROS 2 ベース Gym 環境 | `simulation_core` |
| `ros2_ws/src/coordinated_navigation/src/` | 旧 ROS ノード | `node_2.py`, `policy_controller_3.py` |
| `run_gym.py`, `check_space.py` | 環境チェック用（廃止済みの `simulation_env-v0` を参照） | — |
| `memo.txt` | 開発初期のメモ | — |
