import toml

from .utils import singleton

@singleton
class TOMLSettingsLoader:
    def __init__(self) -> None:
        self.path = "settings.toml"
        self._read()

    def _read(self):
        with open(self.path, 'r') as f:
            self.settings = toml.load(f)
    
    def __getitem__(self, key:str):
        return self.settings[key]
    
    def __repr__(self) -> str:
        return str(self.settings)