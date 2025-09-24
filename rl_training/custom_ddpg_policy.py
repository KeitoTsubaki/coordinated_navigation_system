from stable_baselines3.td3.policies import TD3Policy
from lidar_network import LiDARNetwork
from actor import CustomActor
from critic import CustomCritic


class CustomDDPGPolicy(TD3Policy):
    def __init__(self, *args, **kwargs):
        self.shared_lidar_network = None
        super().__init__(*args, **kwargs)

    def _get_shared_lidar_network(self):
        if self.shared_lidar_network is None:
            self.shared_lidar_network = LiDARNetwork()
        return self.shared_lidar_network

    def _build(self, lr_schedule):
        super()._build(lr_schedule)

    """
    make_actorとmake_criticをオーバーライドすると、自分でactor_targetやactor_optimizerを定義する必要がなくなるらしい
    自動でやってくれるらしい
    """

    def make_actor(self, features_extractor=None):
        actor = CustomActor(
            self.observation_space, self.action_space, self._get_shared_lidar_network()
        )
        return actor.to(self.device)

    def make_critic(self, features_extractor=None):
        critic = CustomCritic(
            self.observation_space, self.action_space, self._get_shared_lidar_network()
        )
        return critic.to(self.device)
