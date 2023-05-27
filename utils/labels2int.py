import json
import pathlib


class Label2Int:
    def __init__(self):
        self.mapping = self._load_mapping()
        self.int2label = {v: k for k, v in self.mapping.items()}

    def _load_mapping(self):
        """Loads the mapping from the json file."""
        path2mapping = pathlib.Path("obligatory3/data/str2int.json")
        if not path2mapping.exists():
            path2mapping = pathlib.Path("data/str2int.json")
        if not path2mapping.exists():
            path2mapping = pathlib.Path("str2int.json")
        if not path2mapping.exists():
            raise FileNotFoundError(
                "Could not find str2int.json. Should be in obligatory3/data/ or data/ or in the root directory."
            )
        return json.load(open(path2mapping))
        
    def __call__(self, str_label) -> int:
        """Converts the string label to an integer."""
        return self.mapping[str_label]

def make_mapping():
    pass

if __name__ == "__main__":
    make_mapping()