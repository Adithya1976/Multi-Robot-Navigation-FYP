from numpy import save
from env import VOEnv

class EnvHandler:
    def __init__(self, world_name):
        self.world_name = world_name
        self.train_env = VOEnv(world_name, display=False)

    def get_train_env(self) -> VOEnv:
        return self.train_env
    
    def get_save_ani_env(self) -> VOEnv:
        self.save_ani_env = VOEnv(self.world_name, display=False, save_ani=True)
        return self.save_ani_env

    def end_save_ani_env(self, name: str):
        suffix = "_" + name + ".mp4"
        self.save_ani_env.end(suffix=suffix)
        del self.save_ani_env