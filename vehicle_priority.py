"""
vehicle_priority.py - Car model priority weights for crossroad management.

Maps VMMR classifier output labels to a numeric priority weight.

Weight meaning:
    > 1.0  = higher priority (slow to accelerate, large, emergency)
    = 1.0  = baseline (standard car)
    < 1.0  = lower priority (fast, small, low road impact)

Add or edit entries freely. The lookup is case-insensitive and tries:
    1. Exact match on full model string
    2. Partial match on any keyword in the model string
    3. Falls back to DEFAULT_WEIGHT = 1.0
"""

from __future__ import annotations

# ── Full model name → weight ───────────────────────────────────────────────────
# Keys must match VMMR classifier class labels exactly (case-insensitive).
# Covers all 196 Stanford Cars classes + common additions.

MODEL_WEIGHTS: dict[str, float] = {
    # ── Emergency / high priority ─────────────────────────────────────────
    "ambulance":                                        5.0,
    "fire truck":                                       5.0,
    "police":                                           4.0,

    # ── Buses ─────────────────────────────────────────────────────────────
    "chevrolet express van 2007":                       2.8,
    "chevrolet express cargo van 2007":                 2.8,
    "ford e-series wagon van 2012":                     2.8,
    "gmc savana van 2012":                              2.8,
    "nissan nv passenger van 2012":                     2.8,
    "ram c-v cargo van minivan 2012":                   2.5,
    "dodge sprinter cargo van 2009":                    2.5,
    "mercedes-benz sprinter van 2012":                  2.5,
    "honda odyssey minivan 2012":                       1.6,
    "honda odyssey minivan 2007":                       1.6,
    "ford freestar minivan 2007":                       1.6,
    "chrysler town and country minivan 2012":           1.6,
    "dodge caravan minivan 1997":                       1.6,

    # ── Pickup trucks ─────────────────────────────────────────────────────
    "ford f-450 super duty crew cab 2012":              3.0,
    "chevrolet silverado 2500hd regular cab 2012":      2.8,
    "dodge ram pickup 3500 crew cab 2010":              2.8,
    "dodge ram pickup 3500 quad cab 2009":              2.8,
    "ford f-150 regular cab 2012":                      2.5,
    "ford f-150 regular cab 2007":                      2.5,
    "chevrolet silverado 1500 extended cab 2012":       2.4,
    "chevrolet silverado 1500 regular cab 2012":        2.4,
    "chevrolet silverado 1500 classic extended cab 2007": 2.4,
    "chevrolet silverado 1500 hybrid crew cab 2012":    2.4,
    "chevrolet avalanche crew cab 2012":                2.4,
    "dodge dakota crew cab 2010":                       2.2,
    "dodge dakota club cab 2007":                       2.2,
    "ford ranger supercab 2011":                        2.2,
    "gmc canyon extended cab 2012":                     2.2,
    "hummer h2 sut crew cab 2009":                      2.2,
    "hummer h3t crew cab 2010":                         2.2,

    # ── Large SUVs ────────────────────────────────────────────────────────
    "chevrolet tahoe hybrid suv 2012":                  2.0,
    "chevrolet traverse suv 2012":                      2.0,
    "ford expedition el suv 2009":                      2.0,
    "gmc yukon hybrid suv 2012":                        2.0,
    "gmc acadia suv 2012":                              1.8,
    "am general hummer suv 2000":                       2.2,
    "land rover range rover suv 2012":                  2.0,
    "land rover lr2 suv 2012":                          1.8,
    "infiniti qx56 suv 2011":                           2.0,
    "cadillac escalade ext crew cab 2007":              2.2,
    "cadillac srx suv 2012":                            1.8,
    "isuzu ascender suv 2008":                          1.8,
    "toyota sequoia suv 2012":                          2.0,
    "toyota 4runner suv 2012":                          1.8,
    "hyundai veracruz suv 2012":                        1.8,
    "hyundai santa fe suv 2012":                        1.8,
    "jeep grand cherokee suv 2012":                     1.8,
    "bmw x5 suv 2007":                                  1.8,
    "bmw x6 suv 2012":                                  1.8,
    "bmw x3 suv 2012":                                  1.6,
    "dodge durango suv 2012":                           1.8,
    "dodge durango suv 2007":                           1.8,
    "dodge journey suv 2012":                           1.6,
    "chrysler aspen suv 2009":                          1.8,
    "mazda tribute suv 2011":                           1.5,
    "volvo xc90 suv 2007":                              1.8,
    "buick enclave suv 2012":                           1.8,
    "buick rainier suv 2007":                           1.8,
    "gmc terrain suv 2012":                             1.5,
    "ford edge suv 2012":                               1.6,
    "jeep wrangler suv 2012":                           1.6,
    "jeep liberty suv 2012":                            1.6,
    "jeep patriot suv 2012":                            1.5,
    "jeep compass suv 2012":                            1.5,
    "hyundai tucson suv 2012":                          1.5,

    # ── Standard sedans / baseline ────────────────────────────────────────
    "toyota camry sedan 2012":                          1.0,
    "toyota corolla sedan 2012":                        1.0,
    "honda accord sedan 2012":                          1.0,
    "honda accord coupe 2012":                          1.0,
    "bmw 3 series sedan 2012":                          1.0,
    "bmw 3 series wagon 2012":                          1.0,
    "audi a5 coupe 2012":                               1.0,
    "audi s4 sedan 2012":                               1.0,
    "audi s4 sedan 2007":                               1.0,
    "audi s6 sedan 2011":                               1.0,
    "hyundai sonata sedan 2012":                        1.0,
    "hyundai elantra sedan 2007":                       1.0,
    "hyundai genesis sedan 2012":                       1.0,
    "kia optima sedan 2012":                            1.0,
    "lincoln town car sedan 2011":                      1.1,
    "cadillac cts-v sedan 2012":                        1.0,
    "dodge charger sedan 2012":                         1.0,
    "chrysler 300 srt-8 2010":                          1.0,
    "volkswagen golf hatchback 2012":                   1.0,
    "volkswagen golf hatchback 1991":                   1.0,
    "volkswagen beetle hatchback 2012":                 1.0,
    "volvo c30 hatchback 2012":                         1.0,
    "volvo 240 sedan 1993":                             1.0,
    "subaru impreza sedan 2012":                        1.0,
    "nissan leaf hatchback 2012":                       1.0,
    "nissan juke hatchback 2012":                       1.0,
    "tesla model s sedan 2012":                         1.0,
    "ford focus sedan 2007":                            1.0,
    "ford fiesta sedan 2012":                           1.0,
    "chevrolet malibu sedan 2007":                      1.0,
    "chevrolet malibu hybrid sedan 2010":               1.0,
    "chevrolet sonic sedan 2012":                       1.0,
    "chevrolet impala sedan 2007":                      1.1,
    "dodge caliber wagon 2012":                         1.0,
    "dodge caliber wagon 2007":                         1.0,
    "suzuki aerio sedan 2007":                          1.0,
    "suzuki kizashi sedan 2012":                        1.0,
    "suzuki sx4 sedan 2012":                            1.0,
    "suzuki sx4 hatchback 2012":                        1.0,
    "hyundai accent sedan 2012":                        1.0,
    "hyundai azera sedan 2012":                         1.0,
    "geo metro convertible 1993":                       0.9,
    "daewoo nubira wagon 2002":                         0.9,
    "scion xd hatchback 2012":                          0.9,
    "smart fortwo hatchback 2012":                      0.8,
    "smart fortwo convertible 2012":                    0.8,
    "fiat 500 abarth 2012":                             0.85,
    "fiat 500 convertible 2012":                        0.85,
    "mini cooper roadster convertible 2012":            0.85,
    "mitsubishi lancer sedan 2012":                     1.0,
    "eagle talon hatchback 1998":                       0.9,
    "plymouth neon coupe 1999":                         0.9,

    # ── Sports / performance — fast, responsive, low priority ─────────────
    "bugatti veyron 16.4 coupe 2009":                   0.6,
    "bugatti veyron 16.4 convertible 2009":             0.6,
    "lamborghini aventador coupe 2012":                 0.65,
    "lamborghini gallardo lp 570-4 superleggera 2012":  0.65,
    "lamborghini reventon coupe 2008":                  0.65,
    "lamborghini diablo coupe 2001":                    0.65,
    "ferrari california convertible 2012":              0.65,
    "ferrari 458 italia coupe 2012":                    0.65,
    "ferrari 458 italia convertible 2012":              0.65,
    "ferrari ff coupe 2012":                            0.65,
    "ferrari cayenne coupe 2012":                       0.65,
    "mclaren mp4-12c coupe 2012":                       0.65,
    "spyker c8 coupe 2009":                             0.7,
    "spyker c8 convertible 2009":                       0.7,
    "aston martin v8 vantage coupe 2012":               0.7,
    "aston martin v8 vantage convertible 2012":         0.7,
    "aston martin virage coupe 2012":                   0.7,
    "aston martin virage convertible 2012":             0.7,
    "bmw m3 coupe 2012":                                0.75,
    "bmw m5 sedan 2010":                                0.75,
    "bmw m6 convertible 2010":                          0.75,
    "bmw z4 convertible 2012":                          0.75,
    "audi r8 coupe 2012":                               0.75,
    "audi rs 4 convertible 2008":                       0.75,
    "audi tts coupe 2012":                              0.8,
    "audi tt rs coupe 2012":                            0.8,
    "audi tt hatchback 2011":                           0.8,
    "ford gt coupe 2006":                               0.7,
    "ford mustang convertible 2007":                    0.8,
    "chevrolet corvette convertible 2012":              0.7,
    "chevrolet corvette zr1 2012":                      0.7,
    "chevrolet corvette ron fellows edition z06 2007":  0.7,
    "chevrolet camaro convertible 2012":                0.8,
    "chevrolet cobalt ss 2010":                         0.8,
    "chevrolet hhr ss 2010":                            0.85,
    "dodge charger srt-8 2009":                         0.8,
    "dodge challenger srt8 2011":                       0.8,
    "chrysler crossfire convertible 2008":              0.8,
    "nissan 240sx coupe 1998":                          0.8,
    "porsche panamera sedan 2012":                      0.75,

    # ── Luxury / limousines — heavier but manageable ──────────────────────
    "rolls-royce phantom sedan 2012":                   1.3,
    "rolls-royce ghost sedan 2012":                     1.3,
    "rolls-royce phantom drophead coupe convertible 2012": 1.2,
    "maybach landaulet convertible 2012":               1.3,
    "bentley mulsanne sedan 2011":                      1.2,
    "bentley continental gt coupe 2012":                1.1,
    "bentley continental gt coupe 2007":                1.1,
    "bentley continental flying spur sedan 2007":       1.2,
    "bentley continental supersports conv. convertible 2012": 1.0,
    "bentley arnage sedan 2009":                        1.2,
    "mercedes-benz s-class sedan 2012":                 1.2,
    "mercedes-benz e-class sedan 2012":                 1.1,
    "mercedes-benz c-class sedan 2012":                 1.0,
    "jaguar xk xkr 2012":                              1.0,
    "fisker karma sedan 2012":                          1.0,
}

# ── Keyword fallback weights ───────────────────────────────────────────────────
# Applied when no exact match found. Checked in order, first match wins.

KEYWORD_WEIGHTS: list[tuple[str, float, str]] = [
    ("ambulance",   5.0, "emergency vehicle"),
    ("fire truck",  5.0, "emergency vehicle"),
    ("firetruck",   5.0, "emergency vehicle"),
    ("police",      4.0, "emergency vehicle"),
    ("bus",         2.8, "large bus"),
    ("truck",       2.5, "truck — slow to accelerate"),
    ("pickup",      2.2, "pickup truck"),
    ("sprinter",    2.5, "cargo van"),
    ("van",         2.0, "van"),
    ("minivan",     1.6, "minivan"),
    ("suv",         1.6, "SUV"),
    ("wagon",       1.2, "station wagon"),
    ("hatchback",   0.95,"compact hatchback"),
    ("convertible", 0.9, "convertible — fast"),
    ("coupe",       0.85,"coupe — sporty"),
    ("sedan",       1.0, "standard sedan"),
    ("motorcycle",  0.7, "motorcycle — fast, low impact"),
    ("motor",       0.7, "motorcycle"),
]

DEFAULT_WEIGHT = 1.0
DEFAULT_REASON = "standard vehicle"


# ── Lookup function ───────────────────────────────────────────────────────────

def get_priority(model_name: str) -> tuple[float, str]:
    """
    Return (weight, reason) for a given VMMR model label.

    Lookup order:
        1. Exact match in MODEL_WEIGHTS (case-insensitive)
        2. Keyword match in KEYWORD_WEIGHTS
        3. DEFAULT_WEIGHT = 1.0

    Examples:
        get_priority("Ford F-450 Super Duty Crew Cab 2012")
        -> (3.0, "Ford F-450 Super Duty Crew Cab 2012")

        get_priority("BMW M3 Coupe 2012")
        -> (0.75, "BMW M3 Coupe 2012")

        get_priority("car")
        -> (1.0, "standard vehicle")
    """
    name = model_name.strip().lower()

    # 1. Exact match
    if name in MODEL_WEIGHTS:
        return MODEL_WEIGHTS[name], model_name

    # 2. Keyword fallback
    for keyword, weight, reason in KEYWORD_WEIGHTS:
        if keyword in name:
            return weight, reason

    return DEFAULT_WEIGHT, DEFAULT_REASON