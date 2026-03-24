from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class PropertyTable:
    T: np.ndarray          # Temperature [K]
    values: np.ndarray     # Property values
    property_name: str     # "Cp", "k", "rho", "alpha"
    unit: str              # "J/(kg·K)", "W/(m·K)", etc.

    def interpolate(self, T_query):
        return np.interp(T_query, self.T, self.values)


@dataclass
class MaterialData:
    name: str              # "Air", "CMSX-4"
    source: str            # "Park", "Lemmon", "Quested"
    properties: Dict[str, PropertyTable] = field(default_factory=dict)

    def get(self, prop_name: str, T):
        return self.properties[prop_name].interpolate(T)


# ── Air Cp data ──────────────────────────────────────────────────

Temp_park = np.linspace(100, 2000, 20)
Cp_park = np.array([
    1.0019e3, 1.0022e3, 1.0045e3, 1.0131e3, 1.0292e3,
    1.0506e3, 1.0745e3, 1.0982e3, 1.1204e3, 1.1405e3,
    1.1583e3, 1.1739e3, 1.1876e3, 1.1997e3, 1.2104e3,
    1.2199e3, 1.2284e3, 1.2361e3, 1.2431e3, 1.2495e3,
])

Temp_lemmon = np.array([
    200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
    300, 310, 320, 330, 340, 350, 360, 370, 380, 390,
    400, 450, 500, 550, 600, 650, 700, 750, 800, 900,
    1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
], dtype=float)
Cp_lemmon = np.array([
    1006.698, 1006.352, 1006.007, 1005.662, 1005.662, 1005.662,
    1005.662, 1005.662, 1005.662, 1006.007, 1006.352, 1006.698,
    1007.388, 1007.733, 1008.424, 1009.114, 1010.15, 1010.84,
    1011.876, 1012.912, 1014.293, 1021.197, 1029.828, 1040.185,
    1051.232, 1062.97, 1075.054, 1086.791, 1098.529, 1120.969,
    1140.993, 1158.945, 1174.48, 1188.29, 1200.373, 1211.075,
    1220.396, 1229.027, 1236.622, 1243.872, 1250.086,
])

AIR_PARK = MaterialData(
    name="Air", source="Park",
    properties={
        "Cp": PropertyTable(T=Temp_park, values=Cp_park,
                            property_name="Cp", unit="J/(kg·K)"),
    },
)

AIR_LEMMON = MaterialData(
    name="Air", source="Lemmon",
    properties={
        "Cp": PropertyTable(T=Temp_lemmon, values=Cp_lemmon,
                            property_name="Cp", unit="J/(kg·K)"),
    },
)

# ── Air thermal conductivity ─────────────────────────────────────

Temp_Incropera = np.array([
    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
    750, 800, 850, 900, 950, 1000, 1050, 1200, 1300, 1400, 1500,
    1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 3000,
], dtype=float)
k_Incropera = np.array([
    9.34, 13.8, 18.1, 22.3, 26.3, 30, 33.8, 37.3, 40.7, 43.9, 46.9,
    49.7, 52.4, 54.9, 57.3, 59.6, 62, 64.3, 66.7, 71.5, 76.3, 82,
    91, 100, 106, 113, 120, 128, 137, 147, 160, 175, 196, 222, 486,
]) * 1e-3  # mW/(m·K) -> W/(m·K)

Temp_Kadoya = np.array([
    85, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450,
    500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100,
    1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
], dtype=float)
k_Kadoya = np.array([
    8.096, 8.480, 9.405, 11.33, 13.18, 14.95, 16.67, 18.36, 22.41,
    26.23, 29.84, 33.28, 36.56, 39.71, 42.77, 45.73, 48.63, 51.46,
    54.25, 56.99, 59.69, 62.37, 65.01, 67.63, 72.81, 77.92, 82.97,
    87.98, 92.96, 97.90, 102.8, 107.7, 112.6, 117.5,
]) * 1e-3  # mW/(m·K) -> W/(m·K)

AIR_INCROPERA = MaterialData(
    name="Air", source="Incropera",
    properties={
        "k": PropertyTable(T=Temp_Incropera, values=k_Incropera,
                           property_name="k", unit="W/(m·K)"),
    },
)

AIR_KADOYA = MaterialData(
    name="Air", source="Kadoya",
    properties={
        "k": PropertyTable(T=Temp_Kadoya, values=k_Kadoya,
                           property_name="k", unit="W/(m·K)"),
    },
)

# ── CMSX-4 (Quested) ─────────────────────────────────────────────

Temp_Quested = np.array([298, 400, 600, 800, 1000, 1200, 1400, 1593, 1653, 1700], dtype=float)

CMSX4_QUESTED = MaterialData(
    name="CMSX-4", source="Quested",
    properties={
        "rho": PropertyTable(
            T=Temp_Quested,
            values=np.array([8700, 8652, 8559, 8466, 8374, 8283, 8193, 8107, 7754, 7710], dtype=float),
            property_name="rho", unit="kg/m^3",
        ),
        "Cp": PropertyTable(
            T=Temp_Quested,
            values=np.array([397, 419, 448, 471, 540, 650, 925, 0, 630, 630], dtype=float),
            property_name="Cp", unit="J/(kg·K)",
        ),
        "k": PropertyTable(
            T=Temp_Quested,
            values=np.array([0, 9, 12, 15.4, 19.9, 21.9, 24.4, 0, 0, 25], dtype=float),
            property_name="k", unit="W/(m·K)",
        ),
        "alpha": PropertyTable(
            T=Temp_Quested,
            values=np.array([0, 2.5, 3.1, 3.8, 4.35, 4.7, 6.1, 0, 4.9, 4.9], dtype=float),
            property_name="alpha", unit="m^2/s (x1e-6)",
        ),
    },
)

# Quested dataset 2 (higher resolution Cp)
Temp_Quested2 = np.array([
    371.56, 469.89, 575, 673.33, 775.06, 872.7, 972.37, 1072.04,
    1172.63, 1274.26, 1375.85, 1475.63, 1596.21, 1674.18, 1774.16, 1874.85,
], dtype=float)
Cp_CMSX4_Quested2 = np.array([
    411.75, 430.71, 441.98, 454.28, 463.51, 487.58, 529.56, 568.99,
    628.33, 709.59, 848.87, 1148.68, 995.26, 673.56, 672.93, 673.93,
], dtype=float)

# Quested dataset 3 (melting region Cp)
Temp_Quested3 = np.array([
    1481.807, 1497.611, 1516.385, 1532.544, 1543.952, 1553.456, 1560.94,
    1568.603, 1579.117, 1585.831, 1593.085, 1599.573, 1603.27, 1606.255,
    1609.595, 1612.089, 1614.345, 1617.439, 1623.52, 1630.243, 1636.312,
    1640.481, 1644.932, 1648.244, 1650.956, 1653.662, 1655.163, 1656.351,
    1658.529, 1659.67, 1660.976, 1661.99, 1663.761, 1666.289, 1669.728,
    1679.907, 1689.938, 1699.714, 1710.408, 1718.725,
], dtype=float)
Cp_CMSX4_Quested3 = np.array([
    837, 858.1, 889.9, 878.4, 921.3, 888.4, 839.4, 817.4, 798, 816.7,
    965.4, 1306.5, 1672, 2045.7, 2414, 2365.2, 2343.4, 2462.5, 3020.3,
    3775.5, 4686.9, 5422.4, 6187.2, 6685.4, 7149.5, 7456.5, 7545.2,
    7292.6, 5483.6, 3975, 2882.8, 1975, 1285.5, 834.9, 725.5, 725.1,
    745.2, 725.6, 714.3, 714,
], dtype=float)
