"""
Cosmetic Chemistry Atom Types

This module provides Python implementations of the cosmetic chemistry atom types
defined in the atom_types.script file. These classes can be used when OpenCog
is not available or for standalone testing.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AtomProperties:
    """Properties associated with cosmetic atoms"""
    ph_range: Optional[Tuple[float, float]] = None
    max_concentration: Optional[float] = None
    allergen_status: bool = False
    cost_per_kg: Optional[float] = None
    stability_temperature: Optional[float] = None
    solubility: Optional[str] = None


class CosmeticAtom:
    """Base class for all cosmetic atoms"""
    
    def __init__(self, name: str, atom_type: str, properties: Optional[AtomProperties] = None):
        self.name = name
        self.atom_type = atom_type
        self.properties = properties or AtomProperties()
    
    def __str__(self):
        return f"{self.atom_type}('{self.name}')"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, CosmeticAtom):
            return False
        return self.name == other.name and self.atom_type == other.atom_type
    
    def __hash__(self):
        return hash((self.name, self.atom_type))


# Ingredient Category Classes
class ACTIVE_INGREDIENT(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "ACTIVE_INGREDIENT", properties)


class PRESERVATIVE(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "PRESERVATIVE", properties)


class EMULSIFIER(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "EMULSIFIER", properties)


class HUMECTANT(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "HUMECTANT", properties)


class SURFACTANT(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "SURFACTANT", properties)


class THICKENER(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "THICKENER", properties)


class EMOLLIENT(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "EMOLLIENT", properties)


class ANTIOXIDANT(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "ANTIOXIDANT", properties)


class UV_FILTER(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "UV_FILTER", properties)


class FRAGRANCE(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "FRAGRANCE", properties)


class COLORANT(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "COLORANT", properties)


class PH_ADJUSTER(CosmeticAtom):
    def __init__(self, name: str, properties: Optional[AtomProperties] = None):
        super().__init__(name, "PH_ADJUSTER", properties)


# Formulation Classes
class CosmeticFormulation:
    """Base class for cosmetic formulations"""
    
    def __init__(self, formulation_type: str, *ingredients: CosmeticAtom):
        self.formulation_type = formulation_type
        self.ingredients = list(ingredients)
        self.properties = {}
    
    def add_ingredient(self, ingredient: CosmeticAtom):
        self.ingredients.append(ingredient)
    
    def get_ingredients_by_type(self, atom_type: str) -> List[CosmeticAtom]:
        return [ing for ing in self.ingredients if ing.atom_type == atom_type]
    
    def __str__(self):
        return f"{self.formulation_type}({len(self.ingredients)} ingredients)"


class SKINCARE_FORMULATION(CosmeticFormulation):
    def __init__(self, *ingredients: CosmeticAtom):
        super().__init__("SKINCARE_FORMULATION", *ingredients)


class HAIRCARE_FORMULATION(CosmeticFormulation):
    def __init__(self, *ingredients: CosmeticAtom):
        super().__init__("HAIRCARE_FORMULATION", *ingredients)


class MAKEUP_FORMULATION(CosmeticFormulation):
    def __init__(self, *ingredients: CosmeticAtom):
        super().__init__("MAKEUP_FORMULATION", *ingredients)


class FRAGRANCE_FORMULATION(CosmeticFormulation):
    def __init__(self, *ingredients: CosmeticAtom):
        super().__init__("FRAGRANCE_FORMULATION", *ingredients)


# Link Classes
class CosmeticLink:
    """Base class for cosmetic links"""
    
    def __init__(self, link_type: str, ingredient1: CosmeticAtom, ingredient2: CosmeticAtom):
        self.link_type = link_type
        self.ingredient1 = ingredient1
        self.ingredient2 = ingredient2
    
    def __str__(self):
        return f"{self.link_type}: {self.ingredient1.name} <-> {self.ingredient2.name}"


class COMPATIBILITY_LINK(CosmeticLink):
    def __init__(self, ingredient1: CosmeticAtom, ingredient2: CosmeticAtom):
        super().__init__("COMPATIBLE", ingredient1, ingredient2)


class INCOMPATIBILITY_LINK(CosmeticLink):
    def __init__(self, ingredient1: CosmeticAtom, ingredient2: CosmeticAtom):
        super().__init__("INCOMPATIBLE", ingredient1, ingredient2)


class SYNERGY_LINK(CosmeticLink):
    def __init__(self, ingredient1: CosmeticAtom, ingredient2: CosmeticAtom):
        super().__init__("SYNERGY", ingredient1, ingredient2)


class ANTAGONISM_LINK(CosmeticLink):
    def __init__(self, ingredient1: CosmeticAtom, ingredient2: CosmeticAtom):
        super().__init__("ANTAGONISM", ingredient1, ingredient2)


# Property Classes
class PropertyType:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value
    
    def __str__(self):
        return f"{self.name}: {self.value}"


class PH_PROPERTY(PropertyType):
    def __init__(self, value: float):
        super().__init__("pH", value)


class VISCOSITY_PROPERTY(PropertyType):
    def __init__(self, value: str):
        super().__init__("Viscosity", value)


class STABILITY_PROPERTY(PropertyType):
    def __init__(self, value: float):
        super().__init__("Stability", value)


class TEXTURE_PROPERTY(PropertyType):
    def __init__(self, value: str):
        super().__init__("Texture", value)


class SPF_PROPERTY(PropertyType):
    def __init__(self, value: int):
        super().__init__("SPF", value)


# Utility functions
def create_ingredient_database() -> Dict[str, CosmeticAtom]:
    """Create a database of common cosmetic ingredients"""
    
    ingredients = {}
    
    # Active ingredients with properties
    ingredients['hyaluronic_acid'] = ACTIVE_INGREDIENT(
        'hyaluronic_acid',
        AtomProperties(ph_range=(5.0, 7.0), max_concentration=2.0, cost_per_kg=500.0)
    )
    
    ingredients['niacinamide'] = ACTIVE_INGREDIENT(
        'niacinamide',
        AtomProperties(ph_range=(5.0, 7.0), max_concentration=10.0, cost_per_kg=80.0)
    )
    
    ingredients['vitamin_c'] = ACTIVE_INGREDIENT(
        'vitamin_c',
        AtomProperties(ph_range=(3.0, 4.0), max_concentration=20.0, cost_per_kg=150.0)
    )
    
    ingredients['retinol'] = ACTIVE_INGREDIENT(
        'retinol',
        AtomProperties(ph_range=(5.5, 6.5), max_concentration=1.0, cost_per_kg=2000.0)
    )
    
    # Supporting ingredients
    ingredients['glycerin'] = HUMECTANT(
        'glycerin',
        AtomProperties(ph_range=(4.0, 8.0), max_concentration=50.0, cost_per_kg=2.0)
    )
    
    ingredients['phenoxyethanol'] = PRESERVATIVE(
        'phenoxyethanol',
        AtomProperties(ph_range=(4.0, 8.0), max_concentration=1.0, cost_per_kg=15.0)
    )
    
    ingredients['cetyl_alcohol'] = EMULSIFIER(
        'cetyl_alcohol',
        AtomProperties(ph_range=(5.0, 8.0), max_concentration=10.0, cost_per_kg=5.0)
    )
    
    ingredients['vitamin_e'] = ANTIOXIDANT(
        'vitamin_e',
        AtomProperties(ph_range=(5.0, 8.0), max_concentration=1.0, cost_per_kg=25.0)
    )
    
    return ingredients


# Known compatibility relationships
KNOWN_COMPATIBLE_PAIRS = [
    ('hyaluronic_acid', 'niacinamide'),
    ('hyaluronic_acid', 'glycerin'),
    ('vitamin_c', 'vitamin_e'),
]

KNOWN_INCOMPATIBLE_PAIRS = [
    ('vitamin_c', 'retinol'),
    ('retinol', 'salicylic_acid'),
]

KNOWN_SYNERGISTIC_PAIRS = [
    ('vitamin_c', 'vitamin_e'),
    ('hyaluronic_acid', 'glycerin'),
]


def check_ingredient_compatibility(ing1: CosmeticAtom, ing2: CosmeticAtom) -> str:
    """Check compatibility between two ingredients"""
    pair = (ing1.name, ing2.name)
    reverse_pair = (ing2.name, ing1.name)
    
    if pair in KNOWN_INCOMPATIBLE_PAIRS or reverse_pair in KNOWN_INCOMPATIBLE_PAIRS:
        return 'incompatible'
    elif pair in KNOWN_SYNERGISTIC_PAIRS or reverse_pair in KNOWN_SYNERGISTIC_PAIRS:
        return 'synergistic'
    elif pair in KNOWN_COMPATIBLE_PAIRS or reverse_pair in KNOWN_COMPATIBLE_PAIRS:
        return 'compatible'
    else:
        return 'unknown'