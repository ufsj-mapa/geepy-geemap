import json

class configParams(object):
    """
    Reads configuration parameters from a JSON file as a dictionary.
    """

    def __init__(self, filename):
        """
        Reads a configuration file.
    
        Args:
            filename: JSON file path
        """
        self.filename = filename

        try:
            with open(self.filename) as f:    
                self.params = json.load(f)
        except ValueError:
            print("JSON Syntax Error!!!!")

    def dump(self):
        """
        Dump dictionary content.
        """
        print(json.dumps(self.params, indent=2, sort_keys=True))
