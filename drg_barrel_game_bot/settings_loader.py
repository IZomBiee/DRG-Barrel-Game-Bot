import toml

from .singleton import singleton

@singleton
class TOMLSettingsLoader:
    def __init__(self) -> None:
        self.path = "settings.toml"
        self._read()
        logic_resolution = (0,
          round(self['display']['resolution'][0]*(self['basket']['axis_line_gap'][0]-(self['basket']['axis_line_gap'][1]-self['basket']['axis_line_gap'][0]))),
          self['display']['resolution'][0],
          round(self['display']['resolution'][0]*(self['basket']['axis_line_gap'][1]-self['basket']['axis_line_gap'][0])))
        self['display']['logic_resolution'] = logic_resolution

    def _read(self):
        with open(self.path, 'r') as f:
            self.settings = toml.load(f)
    
    def __getitem__(self, key:str):
        return self.settings[key]
    
    def __repr__(self) -> str:
        return str(self.settings)