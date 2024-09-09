class ModelStats:
    def __init__(self, name: str, **kwargs: int) -> None:
        """
        Initialize a ModelStats instance with a name and optional attributes.

        :param name: The name of the model.
        :type name: str
        :param kwargs: Optional keyword arguments representing the initial values for the stats attributes.
        :type kwargs: int
        """
        self.name: str = name

        # List of attributes stored in the 'stats' dictionary
        self.stats: list[str] = [
                      "abstract", "abstractNature", "aggregationKind", "AggregationKind", "attribute", "begin",
                      "bringsAbout", "cardinality", "Cardinality", "cardinalityValue", "categorizer", "category",
                      "characterization", "Class", "Classifier", "ClassStereotype", "ClassView", "collective",
                      "collectiveNature", "comparative", "componentOf", "composite", "ConnectorView",
                      "containsModelElement", "containsView", "creation", "datatype", "DecoratableElement",
                      "derivation", "description", "diagram", "Diagram", "DiagramElement", "ElementView", "end",
                      "enumeration", "event", "eventNature", "externalDependence", "extrinsicModeNature",
                      "functionalComplexNature", "general", "generalization", "Generalization", "GeneralizationSet",
                      "GeneralizationSetView", "GeneralizationView", "height", "historicalDependence", "historicalRole",
                      "historicalRoleMixin", "instantiation", "intrinsicModeNature", "isAbstract", "isComplete",
                      "isDerived", "isDisjoint", "isExtensional", "isOrdered", "isPowertype", "isReadOnly", "isViewOf",
                      "kind", "literal", "Literal", "lowerBound", "manifestation", "material", "mediation", "memberOf",
                      "mixin", "mode", "model", "ModelElement", "name", "NodeView", "none", "Note", "NoteView",
                      "OntologicalNature", "OntoumlElement", "order", "owner", "Package", "PackageView",
                      "participation", "participational", "Path", "phase", "phaseMixin", "point", "Point", "project",
                      "Project", "property", "Property", "PropertyStereotype", "propertyType", "quality",
                      "qualityNature", "quantity", "quantityNature", "Rectangle", "RectangularShape",
                      "redefinesProperty", "Relation", "relationEnd", "RelationStereotype", "RelationView", "relator",
                      "relatorNature", "restrictedTo", "role", "roleMixin", "shape", "Shape", "shared", "situation",
                      "situationNature", "sourceEnd", "sourceView", "specific", "stereotype", "Stereotype",
                      "subCollectionOf", "subkind", "subQuantityOf", "subsetsProperty", "targetEnd", "targetView",
                      "termination", "text", "Text", "topLeftPosition", "triggers", "type", "typeNature", "upperBound",
                      "width", "xCoordinate", "yCoordinate"]

        # Initialize 'stats' dictionary with default value 0 for all attributes
        self.stats: dict[str, int] = {attr: 0 for attr in self.stats}

        # Update 'stats' dictionary with any provided keyword arguments
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = int(value)  # Ensure values are stored as integers

