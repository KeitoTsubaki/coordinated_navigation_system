import torch
from stable_baselines3.td3.policies import TD3Policy
from lidar_network import LiDARNetwork
from actor import CustomActor
from critic import CustomCritic


class CustomDDPGPolicy(TD3Policy):
    def __init__(self, *args, **kwargs):
        self.shared_lidar_network = None
        self._custom_lr_schedule = None
        super().__init__(*args, **kwargs)
        
    def _get_shared_lidar_network(self):
        if self.shared_lidar_network is None:
            self.shared_lidar_network = LiDARNetwork()
        return self.shared_lidar_network
    
    def _build(self, lr_schedule):
        self._custom_lr_schedule = lr_schedule
        super()._build(lr_schedule)
        
        
        # --- actor/criticを作り直す ---
        # self.actor = CustomActor(self.shared_lidar_network, self.action_space.shape[-1]).to(self.device)
        # self.actor_target = CustomActor(self.shared_lidar_network, self.action_space.shape[-1]).to(self.device)
        # self.critic = CustomCritic(self.shared_lidar_network, self.action_space.shape[-1]).to(self.device)
        # self.critic_target = CustomCritic(self.shared_lidar_network, self.action_space.shape[-1]).to(self.device)

        # --- optimizerを再定義 ---
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_schedule(1))
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_schedule(1))
        
    """
    make_actorとmake_criticをオーバーライドすると、自分でactor_targetやactor_optimizerを定義する必要がなくなるらしい
    自動でやってくれるらしい
    """
    def make_actor(self, features_extractor = None):
        actor = CustomActor(self.observation_space, self.action_space, self._get_shared_lidar_network(), self._custom_lr_schedule)
        return actor
    
    def make_critic(self, features_extractor = None):
        critic = CustomCritic(self.observation_space, self.action_space, self._get_shared_lidar_network(), self._custom_lr_schedule)
        return critic
        