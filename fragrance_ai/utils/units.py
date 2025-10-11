# fragrance_ai/utils/units.py

from typing import Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Unit Enums
# ============================================================================

class MassUnit(str, Enum):
    """Mass units"""
    GRAM = "g"
    KILOGRAM = "kg"
    MILLIGRAM = "mg"
    POUND = "lb"
    OUNCE = "oz"


class VolumeUnit(str, Enum):
    """Volume units"""
    MILLILITER = "ml"
    LITER = "l"
    MICROLITER = "ul"
    GALLON = "gal"
    FLUID_OUNCE = "fl_oz"


class ConcentrationUnit(str, Enum):
    """Concentration units"""
    PERCENT = "%"
    PPM = "ppm"  # Parts per million
    PPB = "ppb"  # Parts per billion
    GRAM_PER_LITER = "g/L"
    MILLIGRAM_PER_LITER = "mg/L"


# ============================================================================
# Density Database
# ============================================================================

@dataclass
class MaterialDensity:
    """Density information for a material"""
    name: str
    density: float  # g/ml at 20°C
    temperature: float = 20.0  # °C
    notes: Optional[str] = None


class DensityDatabase:
    """Database of material densities for fragrance industry"""

    def __init__(self):
        self.densities: Dict[str, MaterialDensity] = {}
        self._load_default_densities()

    def _load_default_densities(self):
        """Load default density values for common materials"""

        # Solvents
        solvents = [
            MaterialDensity("ethanol", 0.789, notes="95% ethanol"),
            MaterialDensity("ethanol_absolute", 0.785, notes="99.9% ethanol"),
            MaterialDensity("dipropylene_glycol", 1.023, notes="DPG"),
            MaterialDensity("isopropyl_myristate", 0.853, notes="IPM"),
            MaterialDensity("benzyl_benzoate", 1.118, notes="BB"),
            MaterialDensity("diethyl_phthalate", 1.118, notes="DEP"),
            MaterialDensity("water", 0.998),
        ]

        # Essential oils
        essential_oils = [
            MaterialDensity("bergamot_oil", 0.875),
            MaterialDensity("lemon_oil", 0.851),
            MaterialDensity("orange_oil", 0.845),
            MaterialDensity("lavender_oil", 0.885),
            MaterialDensity("rose_oil", 0.855),
            MaterialDensity("jasmine_absolute", 0.945),
            MaterialDensity("sandalwood_oil", 0.973),
            MaterialDensity("patchouli_oil", 0.955),
            MaterialDensity("ylang_ylang_oil", 0.925),
            MaterialDensity("vetiver_oil", 0.988),
        ]

        # Aromatic compounds
        aromatics = [
            MaterialDensity("linalool", 0.858),
            MaterialDensity("limonene", 0.841),
            MaterialDensity("citral", 0.888),
            MaterialDensity("geraniol", 0.879),
            MaterialDensity("eugenol", 1.064),
            MaterialDensity("vanillin", 1.056),
            MaterialDensity("coumarin", 0.935),
            MaterialDensity("benzyl_acetate", 1.055),
            MaterialDensity("phenylethyl_alcohol", 1.017),
            MaterialDensity("iso_e_super", 0.946),
            MaterialDensity("hedione", 0.998),
            MaterialDensity("galaxolide", 1.015),
        ]

        # Add all to database
        for material in solvents + essential_oils + aromatics:
            self.densities[material.name.lower()] = material

    def get_density(self, material_name: str) -> Optional[float]:
        """Get density for a material"""
        return self.densities.get(material_name.lower())

    def add_custom_density(self, material: MaterialDensity):
        """Add custom density to database"""
        self.densities[material.name.lower()] = material


# ============================================================================
# Unit Converter
# ============================================================================

class UnitConverter:
    """Converter for fragrance industry units"""

    def __init__(self):
        self.density_db = DensityDatabase()

    # Mass conversions
    def convert_mass(self, value: float, from_unit: MassUnit, to_unit: MassUnit) -> float:
        """Convert between mass units"""

        # Convert to grams first
        if from_unit == MassUnit.GRAM:
            grams = value
        elif from_unit == MassUnit.KILOGRAM:
            grams = value * 1000
        elif from_unit == MassUnit.MILLIGRAM:
            grams = value / 1000
        elif from_unit == MassUnit.POUND:
            grams = value * 453.592
        elif from_unit == MassUnit.OUNCE:
            grams = value * 28.3495
        else:
            raise ValueError(f"Unknown mass unit: {from_unit}")

        # Convert from grams to target unit
        if to_unit == MassUnit.GRAM:
            return grams
        elif to_unit == MassUnit.KILOGRAM:
            return grams / 1000
        elif to_unit == MassUnit.MILLIGRAM:
            return grams * 1000
        elif to_unit == MassUnit.POUND:
            return grams / 453.592
        elif to_unit == MassUnit.OUNCE:
            return grams / 28.3495
        else:
            raise ValueError(f"Unknown mass unit: {to_unit}")

    # Volume conversions
    def convert_volume(self, value: float, from_unit: VolumeUnit, to_unit: VolumeUnit) -> float:
        """Convert between volume units"""

        # Convert to milliliters first
        if from_unit == VolumeUnit.MILLILITER:
            ml = value
        elif from_unit == VolumeUnit.LITER:
            ml = value * 1000
        elif from_unit == VolumeUnit.MICROLITER:
            ml = value / 1000
        elif from_unit == VolumeUnit.GALLON:
            ml = value * 3785.41
        elif from_unit == VolumeUnit.FLUID_OUNCE:
            ml = value * 29.5735
        else:
            raise ValueError(f"Unknown volume unit: {from_unit}")

        # Convert from milliliters to target unit
        if to_unit == VolumeUnit.MILLILITER:
            return ml
        elif to_unit == VolumeUnit.LITER:
            return ml / 1000
        elif to_unit == VolumeUnit.MICROLITER:
            return ml * 1000
        elif to_unit == VolumeUnit.GALLON:
            return ml / 3785.41
        elif to_unit == VolumeUnit.FLUID_OUNCE:
            return ml / 29.5735
        else:
            raise ValueError(f"Unknown volume unit: {to_unit}")

    # Mass to volume conversions (requires density)
    def mass_to_volume(
        self,
        mass: float,
        mass_unit: MassUnit,
        material_name: str,
        volume_unit: VolumeUnit = VolumeUnit.MILLILITER
    ) -> Optional[float]:
        """Convert mass to volume using density"""

        density_info = self.density_db.get_density(material_name)
        if density_info is None:
            return None

        # Convert mass to grams
        mass_grams = self.convert_mass(mass, mass_unit, MassUnit.GRAM)

        # Calculate volume in ml (density is in g/ml)
        volume_ml = mass_grams / density_info.density

        # Convert to desired volume unit
        return self.convert_volume(volume_ml, VolumeUnit.MILLILITER, volume_unit)

    # Volume to mass conversions (requires density)
    def volume_to_mass(
        self,
        volume: float,
        volume_unit: VolumeUnit,
        material_name: str,
        mass_unit: MassUnit = MassUnit.GRAM
    ) -> Optional[float]:
        """Convert volume to mass using density"""

        density_info = self.density_db.get_density(material_name)
        if density_info is None:
            return None

        # Convert volume to ml
        volume_ml = self.convert_volume(volume, volume_unit, VolumeUnit.MILLILITER)

        # Calculate mass in grams (density is in g/ml)
        mass_grams = volume_ml * density_info.density

        # Convert to desired mass unit
        return self.convert_mass(mass_grams, MassUnit.GRAM, mass_unit)

    # Concentration conversions
    def convert_concentration(
        self,
        value: float,
        from_unit: ConcentrationUnit,
        to_unit: ConcentrationUnit,
        density: float = 1.0
    ) -> float:
        """
        Convert between concentration units

        Args:
            value: Concentration value
            from_unit: Source unit
            to_unit: Target unit
            density: Density for g/L conversions (default 1.0 for water)
        """

        # Convert to percent first
        if from_unit == ConcentrationUnit.PERCENT:
            percent = value
        elif from_unit == ConcentrationUnit.PPM:
            percent = value / 10000
        elif from_unit == ConcentrationUnit.PPB:
            percent = value / 10000000
        elif from_unit == ConcentrationUnit.GRAM_PER_LITER:
            # Assuming density in g/ml, 1L = 1000ml
            percent = (value / (density * 1000)) * 100
        elif from_unit == ConcentrationUnit.MILLIGRAM_PER_LITER:
            percent = (value / (density * 1000000)) * 100
        else:
            raise ValueError(f"Unknown concentration unit: {from_unit}")

        # Convert from percent to target unit
        if to_unit == ConcentrationUnit.PERCENT:
            return percent
        elif to_unit == ConcentrationUnit.PPM:
            return percent * 10000
        elif to_unit == ConcentrationUnit.PPB:
            return percent * 10000000
        elif to_unit == ConcentrationUnit.GRAM_PER_LITER:
            return (percent / 100) * density * 1000
        elif to_unit == ConcentrationUnit.MILLIGRAM_PER_LITER:
            return (percent / 100) * density * 1000000
        else:
            raise ValueError(f"Unknown concentration unit: {to_unit}")

    # Batch size calculations
    def calculate_batch_quantities(
        self,
        formula: Dict[str, float],  # ingredient -> percentage
        batch_size: float,
        batch_unit: Union[MassUnit, VolumeUnit],
        output_unit: Union[MassUnit, VolumeUnit]
    ) -> Dict[str, float]:
        """
        Calculate ingredient quantities for a batch

        Args:
            formula: Dictionary of ingredient names to percentages
            batch_size: Total batch size
            batch_unit: Unit of batch size
            output_unit: Desired output unit for ingredients

        Returns:
            Dictionary of ingredient names to quantities
        """

        # Normalize percentages to sum to 100
        total_percent = sum(formula.values())
        normalized_formula = {
            ing: (pct / total_percent) * 100
            for ing, pct in formula.items()
        }

        quantities = {}

        for ingredient, percent in normalized_formula.items():
            # Calculate quantity based on batch size
            quantity = (percent / 100) * batch_size

            # Convert units if necessary
            if isinstance(batch_unit, MassUnit) and isinstance(output_unit, MassUnit):
                quantities[ingredient] = self.convert_mass(quantity, batch_unit, output_unit)
            elif isinstance(batch_unit, VolumeUnit) and isinstance(output_unit, VolumeUnit):
                quantities[ingredient] = self.convert_volume(quantity, batch_unit, output_unit)
            elif isinstance(batch_unit, MassUnit) and isinstance(output_unit, VolumeUnit):
                # Need density for conversion
                vol = self.mass_to_volume(quantity, batch_unit, ingredient, output_unit)
                if vol is not None:
                    quantities[ingredient] = vol
                else:
                    # Fallback: assume density of 1.0 g/ml
                    mass_g = self.convert_mass(quantity, batch_unit, MassUnit.GRAM)
                    vol_ml = mass_g  # Assuming density = 1.0
                    quantities[ingredient] = self.convert_volume(
                        vol_ml, VolumeUnit.MILLILITER, output_unit
                    )
            elif isinstance(batch_unit, VolumeUnit) and isinstance(output_unit, MassUnit):
                # Need density for conversion
                mass = self.volume_to_mass(quantity, batch_unit, ingredient, output_unit)
                if mass is not None:
                    quantities[ingredient] = mass
                else:
                    # Fallback: assume density of 1.0 g/ml
                    vol_ml = self.convert_volume(quantity, batch_unit, VolumeUnit.MILLILITER)
                    mass_g = vol_ml  # Assuming density = 1.0
                    quantities[ingredient] = self.convert_mass(mass_g, MassUnit.GRAM, output_unit)
            else:
                quantities[ingredient] = quantity

        return quantities


# ============================================================================
# Helper Functions
# ============================================================================

def format_quantity(value: float, unit: Union[MassUnit, VolumeUnit], precision: int = 2) -> str:
    """Format quantity with unit for display"""
    return f"{value:.{precision}f} {unit.value}"


def parse_quantity(quantity_str: str) -> tuple[float, str]:
    """
    Parse a quantity string like "10.5 g" or "250 ml"

    Returns:
        Tuple of (value, unit)
    """
    parts = quantity_str.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid quantity format: {quantity_str}")

    try:
        value = float(parts[0])
    except ValueError:
        raise ValueError(f"Invalid number in quantity: {parts[0]}")

    unit = parts[1].lower()
    return value, unit


def calculate_dilution(
    concentrate_percent: float,
    target_percent: float,
    concentrate_volume: float = 100.0
) -> float:
    """
    Calculate dilution required

    Args:
        concentrate_percent: Original concentration (%)
        target_percent: Desired concentration (%)
        concentrate_volume: Volume of concentrate (ml)

    Returns:
        Volume of diluent to add (ml)
    """
    if target_percent >= concentrate_percent:
        return 0.0

    dilution_factor = concentrate_percent / target_percent
    total_volume = concentrate_volume * dilution_factor
    diluent_volume = total_volume - concentrate_volume

    return diluent_volume


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'MassUnit',
    'VolumeUnit',
    'ConcentrationUnit',
    'MaterialDensity',
    'DensityDatabase',
    'UnitConverter',
    'format_quantity',
    'parse_quantity',
    'calculate_dilution'
]