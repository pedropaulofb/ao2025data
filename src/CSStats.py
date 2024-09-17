class CSStats:
    def __init__(self, name: str, **kwargs: int) -> None:
        """
        Initialize a SPOStats instance with a name and optional attributes.

        :param name: The name of the model.
        :type name: str
        :param kwargs: Optional keyword arguments representing the initial values for the stats attributes.
        :type kwargs: int
        """
        self.name: str = name

        # List of attributes stored in the 'stats' dictionary
        self.stats: list[str] = ["abstract", "category", "collective", "datatype", "enumeration", "event",
                                 "historicalRole", "historicalRoleMixin", "kind", "mixin", "mode", "phase",
                                 "phaseMixin", "quality", "quantity", "relator", "role", "roleMixin", "situation",
                                 "subkind", "type", "none", "other"]

        # Initialize 'stats' dictionary with default value 0 for all attributes
        self.stats: dict[str, int] = {attr: 0 for attr in self.stats}

        # Update 'stats' dictionary with any provided keyword arguments
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = int(value)  # Ensure values are stored as integers
