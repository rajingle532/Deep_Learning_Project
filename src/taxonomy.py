"""
Canonical Taxonomy & Harmonization Mapping Module.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Maps 38 PlantVillage classes (Source Domain) and 28 PlantDoc classes (Target Domain)
to 28 unified, canonical class labels with clean display names and agronomic metadata.
"""

from typing import Dict, List, Optional

# 28 Canonical class keys in alphabetical order
CANONICAL_CLASSES: List[str] = [
    "apple_apple_scab",
    "apple_cedar_apple_rust",
    "apple_healthy",
    "bell_pepper_bacterial_spot",
    "bell_pepper_healthy",
    "blueberry_healthy",
    "cherry_healthy",
    "corn_gray_leaf_spot",
    "corn_leaf_blight",
    "corn_rust",
    "grape_black_rot",
    "grape_healthy",
    "peach_healthy",
    "potato_early_blight",
    "potato_late_blight",
    "raspberry_healthy",
    "soybean_healthy",
    "squash_powdery_mildew",
    "strawberry_healthy",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites",
    "tomato_yellow_leaf_curl_virus",
]

NUM_CLASSES = len(CANONICAL_CLASSES)  # 28

CANONICAL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}
ID_TO_CANONICAL: Dict[int, str] = {idx: name for idx, name in enumerate(CANONICAL_CLASSES)}

# PlantVillage exact directory name -> Canonical key
PLANTVILLAGE_TO_CANONICAL: Dict[str, str] = {
    "Apple___Apple_scab": "apple_apple_scab",
    "Apple___Cedar_apple_rust": "apple_cedar_apple_rust",
    "Apple___healthy": "apple_healthy",
    "Pepper,_bell___Bacterial_spot": "bell_pepper_bacterial_spot",
    "Pepper,_bell___healthy": "bell_pepper_healthy",
    "Blueberry___healthy": "blueberry_healthy",
    "Cherry_(including_sour)___healthy": "cherry_healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn_gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight": "corn_leaf_blight",
    "Corn_(maize)___Common_rust_": "corn_rust",
    "Grape___Black_rot": "grape_black_rot",
    "Grape___healthy": "grape_healthy",
    "Peach___healthy": "peach_healthy",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Raspberry___healthy": "raspberry_healthy",
    "Soybean___healthy": "soybean_healthy",
    "Squash___Powdery_mildew": "squash_powdery_mildew",
    "Strawberry___healthy": "strawberry_healthy",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___healthy": "tomato_healthy",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
}

# PlantDoc exact directory name -> Canonical key
PLANTDOC_TO_CANONICAL: Dict[str, str] = {
    "Apple Scab Leaf": "apple_apple_scab",
    "Apple rust leaf": "apple_cedar_apple_rust",
    "Apple leaf": "apple_healthy",
    "Bell_pepper leaf spot": "bell_pepper_bacterial_spot",
    "Bell_pepper leaf": "bell_pepper_healthy",
    "Blueberry leaf": "blueberry_healthy",
    "Cherry leaf": "cherry_healthy",
    "Corn Gray leaf spot": "corn_gray_leaf_spot",
    "Corn leaf blight": "corn_leaf_blight",
    "Corn rust leaf": "corn_rust",
    "grape leaf black rot": "grape_black_rot",
    "grape leaf": "grape_healthy",
    "Peach leaf": "peach_healthy",
    "Potato leaf early blight": "potato_early_blight",
    "Potato leaf late blight": "potato_late_blight",
    "Raspberry leaf": "raspberry_healthy",
    "Soyabean leaf": "soybean_healthy",
    "Squash Powdery mildew leaf": "squash_powdery_mildew",
    "Strawberry leaf": "strawberry_healthy",
    "Tomato leaf bacterial spot": "tomato_bacterial_spot",
    "Tomato Early blight leaf": "tomato_early_blight",
    "Tomato leaf": "tomato_healthy",
    "Tomato leaf late blight": "tomato_late_blight",
    "Tomato mold leaf": "tomato_leaf_mold",
    "Tomato leaf mosaic virus": "tomato_mosaic_virus",
    "Tomato Septoria leaf spot": "tomato_septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "tomato_spider_mites",
    "Tomato leaf yellow virus": "tomato_yellow_leaf_curl_virus",
}

# Human-friendly display names for UI & Evaluation
CANONICAL_DISPLAY_NAMES: Dict[str, str] = {
    "apple_apple_scab": "Apple: Scab",
    "apple_cedar_apple_rust": "Apple: Cedar Rust",
    "apple_healthy": "Apple: Healthy",
    "bell_pepper_bacterial_spot": "Bell Pepper: Bacterial Spot",
    "bell_pepper_healthy": "Bell Pepper: Healthy",
    "blueberry_healthy": "Blueberry: Healthy",
    "cherry_healthy": "Cherry: Healthy",
    "corn_gray_leaf_spot": "Corn: Gray Leaf Spot",
    "corn_leaf_blight": "Corn: Northern Leaf Blight",
    "corn_rust": "Corn: Common Rust",
    "grape_black_rot": "Grape: Black Rot",
    "grape_healthy": "Grape: Healthy",
    "peach_healthy": "Peach: Healthy",
    "potato_early_blight": "Potato: Early Blight",
    "potato_late_blight": "Potato: Late Blight",
    "raspberry_healthy": "Raspberry: Healthy",
    "soybean_healthy": "Soybean: Healthy",
    "squash_powdery_mildew": "Squash: Powdery Mildew",
    "strawberry_healthy": "Strawberry: Healthy",
    "tomato_bacterial_spot": "Tomato: Bacterial Spot",
    "tomato_early_blight": "Tomato: Early Blight",
    "tomato_healthy": "Tomato: Healthy",
    "tomato_late_blight": "Tomato: Late Blight",
    "tomato_leaf_mold": "Tomato: Leaf Mold",
    "tomato_mosaic_virus": "Tomato: Mosaic Virus",
    "tomato_septoria_leaf_spot": "Tomato: Septoria Leaf Spot",
    "tomato_spider_mites": "Tomato: Two-Spotted Spider Mites",
    "tomato_yellow_leaf_curl_virus": "Tomato: Yellow Leaf Curl Virus",
}

# Grouping by crop family
CROP_GROUPS: Dict[str, List[str]] = {
    "Tomato": [c for c in CANONICAL_CLASSES if c.startswith("tomato_")],
    "Apple": [c for c in CANONICAL_CLASSES if c.startswith("apple_")],
    "Corn": [c for c in CANONICAL_CLASSES if c.startswith("corn_")],
    "Potato": [c for c in CANONICAL_CLASSES if c.startswith("potato_")],
    "Bell Pepper": [c for c in CANONICAL_CLASSES if c.startswith("bell_pepper_")],
    "Grape": [c for c in CANONICAL_CLASSES if c.startswith("grape_")],
    "Other Berries & Crops": [
        "blueberry_healthy",
        "cherry_healthy",
        "peach_healthy",
        "raspberry_healthy",
        "soybean_healthy",
        "squash_powdery_mildew",
        "strawberry_healthy",
    ],
}

# Agronomic Treatment & Advisory for Streamlit Web UI
TREATMENT_ADVISORY: Dict[str, Dict[str, str]] = {
    "apple_apple_scab": {
        "pathogen": "Fungus (Venturia inaequalis)",
        "symptoms": "Olive-green to black velvety spots on leaf surface.",
        "organic_treatment": "Apply sulfur spray or neem oil during early bud break; rake and destroy fallen leaves.",
        "chemical_treatment": "Fungicides containing Captan, Mancozeb, or myclobutanil.",
    },
    "apple_cedar_apple_rust": {
        "pathogen": "Fungus (Gymnosporangium juniperi-virginianae)",
        "symptoms": "Bright yellow-orange circular lesions on leaf tops.",
        "organic_treatment": "Remove nearby cedar/juniper galls within 1 mile; use copper-based fungicides.",
        "chemical_treatment": "Immunox or chlorothalonil applied as blossoms open.",
    },
    "apple_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Vibrant green foliage with regular venation and no necrosis.",
        "organic_treatment": "Maintain balanced irrigation, seasonal pruning, and regular compost mulching.",
        "chemical_treatment": "No intervention required.",
    },
    "bell_pepper_bacterial_spot": {
        "pathogen": "Bacterium (Xanthomonas campestris)",
        "symptoms": "Small, water-soaked blister lesions turning brown with yellow halos.",
        "organic_treatment": "Apply copper octanoate; avoid overhead sprinkler irrigation.",
        "chemical_treatment": "Fixed copper bactericides combined with mancozeb.",
    },
    "bell_pepper_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Uniform glossy green foliage without spotting or wilting.",
        "organic_treatment": "Maintain well-draining soil and consistent moisture.",
        "chemical_treatment": "No intervention required.",
    },
    "blueberry_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Healthy dark green, waxy leaves.",
        "organic_treatment": "Maintain acidic soil pH (4.5 - 5.5) with pine bark mulch.",
        "chemical_treatment": "No intervention required.",
    },
    "cherry_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Unblemished ovate leaves with serrated margins.",
        "organic_treatment": "Annual pruning for good airflow and sunlight penetration.",
        "chemical_treatment": "No intervention required.",
    },
    "corn_gray_leaf_spot": {
        "pathogen": "Fungus (Cercospora zeae-maydis)",
        "symptoms": "Rectangular, tan-to-gray lesions restricted by leaf veins.",
        "organic_treatment": "Practice crop rotation with non-host crops and deep tillage of crop debris.",
        "chemical_treatment": "Strobilurin or triazole fungicides applied at VT-R1 growth stage.",
    },
    "corn_leaf_blight": {
        "pathogen": "Fungus (Exserohilum turcicum)",
        "symptoms": "Large, cigar-shaped grayish-green to tan lesions.",
        "organic_treatment": "Plant resistant hybrids; ensure minimum 2-year crop rotation.",
        "chemical_treatment": "Azoxystrobin + propiconazole fungicide application.",
    },
    "corn_rust": {
        "pathogen": "Fungus (Puccinia sorghi)",
        "symptoms": "Golden-brown to cinnamon-brown powdery pustules on both leaf surfaces.",
        "organic_treatment": "Early planting and use of resistant hybrid seed lines.",
        "chemical_treatment": "Pyraclostrobin or tebuconazole upon first pustule appearance.",
    },
    "grape_black_rot": {
        "pathogen": "Fungus (Guignardia bidwellii)",
        "symptoms": "Reddish-brown circular leaf spots with tiny black pycnidia pimples.",
        "organic_treatment": "Prune infected canes; remove mummified berries from vine.",
        "chemical_treatment": "Mancozeb or myclobutanil from early shoot emergence.",
    },
    "grape_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Broad lobed green leaves without marginal discoloration.",
        "organic_treatment": "Canopy management to ensure airflow across grape clusters.",
        "chemical_treatment": "No intervention required.",
    },
    "peach_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Long, lanceolate green foliage free of chlorosis.",
        "organic_treatment": "Proper nitrogen fertilization and dormant oil spraying.",
        "chemical_treatment": "No intervention required.",
    },
    "potato_early_blight": {
        "pathogen": "Fungus (Alternaria solani)",
        "symptoms": "Target-board concentric rings forming dark brown angular spots.",
        "organic_treatment": "Drip irrigation, adequate potassium fertilization, copper sulfate spray.",
        "chemical_treatment": "Chlorothalonil, mancozeb, or azoxystrobin fungicides.",
    },
    "potato_late_blight": {
        "pathogen": "Oomycete (Phytophthora infestans)",
        "symptoms": "Water-soaked dark lesions with white moldy fungal growth on undersides.",
        "organic_treatment": "Destroy infected haulms immediately; avoid excessive moisture.",
        "chemical_treatment": "Systemic fungicides (Metalaxyl, Cymoxanil, or Mandipropamid).",
    },
    "raspberry_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Compound leaves with silvery-white undersides and healthy green tops.",
        "organic_treatment": "Trellising for ventilation; organic compost mulch.",
        "chemical_treatment": "No intervention required.",
    },
    "soybean_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Trifoliate leaves with vibrant, uniform chlorophyll pigmentation.",
        "organic_treatment": "Rhizobium inoculation and routine scouting for insect vectors.",
        "chemical_treatment": "No intervention required.",
    },
    "squash_powdery_mildew": {
        "pathogen": "Fungus (Podosphaera xanthii)",
        "symptoms": "White, talcum powder-like fungal patches across upper leaf surfaces.",
        "organic_treatment": "Potassium bicarbonate spray, dilute milk solution (40%), or sulfur dust.",
        "chemical_treatment": "Triflumizole, myclobutanil, or quinoxyfen.",
    },
    "strawberry_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Glossy, trifoliate leaves with scalloped margins.",
        "organic_treatment": "Straw bedding to keep foliage off bare moist soil.",
        "chemical_treatment": "No intervention required.",
    },
    "tomato_bacterial_spot": {
        "pathogen": "Bacterium (Xanthomonas spp.)",
        "symptoms": "Small greasy dark brown lesions with chlorotic halos.",
        "organic_treatment": "Crop rotation with corn; copper hydroxide sprays.",
        "chemical_treatment": "Streptomycin or fixed copper tank-mixed with mancozeb.",
    },
    "tomato_early_blight": {
        "pathogen": "Fungus (Alternaria solani)",
        "symptoms": "Concentric bullseye-shaped brown patches starting on lower leaves.",
        "organic_treatment": "Mulching at base; prune lower 12 inches of foliage to stop soil splashing.",
        "chemical_treatment": "Chlorothalonil or copper-based preventative fungicide.",
    },
    "tomato_healthy": {
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Deep green compound leaves with healthy trichomes (leaf hairs).",
        "organic_treatment": "Regular staking, consistent watering, and balanced N-P-K feeding.",
        "chemical_treatment": "No intervention required.",
    },
    "tomato_late_blight": {
        "pathogen": "Oomycete (Phytophthora infestans)",
        "symptoms": "Rapidly expanding water-soaked greasy brown lesions turning brittle.",
        "organic_treatment": "Remove and bag infected plants; avoid working with wet foliage.",
        "chemical_treatment": "Preventative chlorothalonil; curative cymoxanil / dimethomorph.",
    },
    "tomato_leaf_mold": {
        "pathogen": "Fungus (Passalora fulva)",
        "symptoms": "Pale green/yellow spots on leaf tops; velvety olive mold on undersides.",
        "organic_treatment": "Lower greenhouse humidity below 85%; increase circulating ventilation.",
        "chemical_treatment": "Copper soap or chlorothalonil.",
    },
    "tomato_mosaic_virus": {
        "pathogen": "Tomato Mosaic Virus (ToMV)",
        "symptoms": "Mottled light and dark green mosaic patterns with fern-like distorted leaves.",
        "organic_treatment": "Wash hands with milk/detergent after smoking; discard infected plants (no cure).",
        "chemical_treatment": "No chemical control available; control aphid and thrip vectors.",
    },
    "tomato_septoria_leaf_spot": {
        "pathogen": "Fungus (Septoria lycopersici)",
        "symptoms": "Numerous small circular spots with grayish-white centers and dark borders.",
        "organic_treatment": "Remove lower affected leaves; drip irrigation only.",
        "chemical_treatment": "Chlorothalonil or copper octanoate applied every 7-10 days.",
    },
    "tomato_spider_mites": {
        "pathogen": "Pest (Tetranychus urticae)",
        "symptoms": "Fine yellow stippling across leaf blades; delicate silken webbing under leaves.",
        "organic_treatment": "Release predatory mites (Phytoseiulus persimilis); insecticidal soap.",
        "chemical_treatment": "Abamectin, bifenazate, or spiromesifen miticides.",
    },
    "tomato_yellow_leaf_curl_virus": {
        "pathogen": "Begomovirus (TYLCV) transmitted by Whitefly",
        "symptoms": "Severe upward leaf curling, stunted growth, and pronounced yellow leaf margins.",
        "organic_treatment": "Use fine insect netting and yellow sticky traps to capture whiteflies.",
        "chemical_treatment": "Imidacloprid or acetamiprid to suppress vector whitefly populations.",
    },
}

