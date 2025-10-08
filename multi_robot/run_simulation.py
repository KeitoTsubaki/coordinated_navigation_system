import pygame
import math
import importlib.resources

from multi_robot_environment import MultiRobotEnvironment


def main():
    num_robots = 2  # ロボットの数

    # simulation_coreパッケージ内の設定ファイルへのパスを取得
    config_path = (
        importlib.resources.files("simulation_core").parent / "config" / "simulator_config.yaml"
    )
    

    env = MultiRobotEnvironment(num_robots=num_robots, render_mode="human", config_path=str(config_path))

    env.reset(map_type="random")

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    env.reset(map_type="random")

        # ここで各ロボットの制御ロジックを実装します
        # 例：最初のロボットをキーボードで操作
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            env.robots[0]["position"] = (env.robots[0]["position"][0], env.robots[0]["position"][1] - 2)
        if keys[pygame.K_LEFT]:
            env.robots[0]["theta"] += 0.05

        env.update_all_sensors()
        env.render_robot_and_lidar()
        pygame.display.update()
        clock.tick(30)  # 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
