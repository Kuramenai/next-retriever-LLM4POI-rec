"""
Three-level POI category taxonomy for session segmentation.

Levels
------
1. Raw category   → the original Foursquare/Gowalla label
2. Mid-level      → semantically coherent grouping (16 buckets)
3. Segmentation   → coarse activity-mode label for phase boundary detection (8 buckets)

"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# Level 1 → Level 2:  Raw category  →  Mid-level taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

RAW_TO_MID: dict[str, str] = {
    # ── Food & Dining ──────────────────────────────────────────────────────
    "African Restaurant": "Food & Dining",
    "American Restaurant": "Food & Dining",
    "Asian Restaurant": "Food & Dining",
    "Australian Restaurant": "Food & Dining",
    "BBQ Joint": "Food & Dining",
    "Bakery": "Food & Dining",
    "Bagel Shop": "Food & Dining",
    "Brazilian Restaurant": "Food & Dining",
    "Breakfast Spot": "Food & Dining",
    "Burger Joint": "Food & Dining",
    "Burrito Place": "Food & Dining",
    "Cajun / Creole Restaurant": "Food & Dining",
    "Caribbean Restaurant": "Food & Dining",
    "Chinese Restaurant": "Food & Dining",
    "Cuban Restaurant": "Food & Dining",
    "Deli / Bodega": "Food & Dining",
    "Dessert Shop": "Food & Dining",
    "Diner": "Food & Dining",
    "Dim Sum Restaurant": "Food & Dining",
    "Donut Shop": "Food & Dining",
    "Dumpling Restaurant": "Food & Dining",
    "Eastern European Restaurant": "Food & Dining",
    "Falafel Restaurant": "Food & Dining",
    "Fast Food Restaurant": "Food & Dining",
    "Filipino Restaurant": "Food & Dining",
    "Fish & Chips Shop": "Food & Dining",
    "Food": "Food & Dining",
    "Food Truck": "Food & Dining",
    "Food & Drink Shop": "Food & Dining",
    "French Restaurant": "Food & Dining",
    "Fried Chicken Joint": "Food & Dining",
    "Gastropub": "Food & Dining",
    "German Restaurant": "Food & Dining",
    "Hot Dog Joint": "Food & Dining",
    "Ice Cream Shop": "Food & Dining",
    "Indian Restaurant": "Food & Dining",
    "Italian Restaurant": "Food & Dining",
    "Japanese Restaurant": "Food & Dining",
    "Korean Restaurant": "Food & Dining",
    "Latin American Restaurant": "Food & Dining",
    "Mac & Cheese Joint": "Food & Dining",
    "Malaysian Restaurant": "Food & Dining",
    "Mediterranean Restaurant": "Food & Dining",
    "Mexican Restaurant": "Food & Dining",
    "Molecular Gastronomy Restaurant": "Food & Dining",
    "Middle Eastern Restaurant": "Food & Dining",
    "Pizza Place": "Food & Dining",
    "Ramen /  Noodle House": "Food & Dining",
    "Restaurant": "Food & Dining",
    "Salad Place": "Food & Dining",
    "Sandwich Place": "Food & Dining",
    "Scandinavian Restaurant": "Food & Dining",
    "Seafood Restaurant": "Food & Dining",
    "Snack Place": "Food & Dining",
    "Soup Place": "Food & Dining",
    "South American Restaurant": "Food & Dining",
    "Southern / Soul Food Restaurant": "Food & Dining",
    "Spanish Restaurant": "Food & Dining",
    "Steakhouse": "Food & Dining",
    "Sushi Restaurant": "Food & Dining",
    "Swiss Restaurant": "Food & Dining",
    "Taco Place": "Food & Dining",
    "Tapas Restaurant": "Food & Dining",
    "Thai Restaurant": "Food & Dining",
    "Vegetarian / Vegan Restaurant": "Food & Dining",
    "Vietnamese Restaurant": "Food & Dining",
    "Wings Joint": "Food & Dining",
    # ── Café & Coffee ─────────────────────────────────────────────────────
    "Café": "Café & Coffee",
    "Coffee Shop": "Café & Coffee",
    "Tea Room": "Café & Coffee",
    # ── Bars & Nightlife ──────────────────────────────────────────────────
    "Bar": "Bars & Nightlife",
    "Beer Garden": "Bars & Nightlife",
    "Brewery": "Bars & Nightlife",
    "Casino": "Bars & Nightlife",
    "Comedy Club": "Bars & Nightlife",
    "Music Venue": "Bars & Nightlife",
    "Nightlife Spot": "Bars & Nightlife",
    "Other Nightlife": "Bars & Nightlife",
    # ── Shopping & Retail ─────────────────────────────────────────────────
    "Antique Shop": "Shopping & Retail",
    "Arts & Crafts Store": "Shopping & Retail",
    "Automotive Shop": "Shopping & Retail",
    "Bike Shop": "Shopping & Retail",
    "Board Shop": "Shopping & Retail",
    "Bookstore": "Shopping & Retail",
    "Bridal Shop": "Shopping & Retail",
    "Bridal Salon": "Shopping & Retail",
    "Camera Store": "Shopping & Retail",
    "Candy Store": "Shopping & Retail",
    "Car Dealership": "Shopping & Retail",
    "Clothing Store": "Shopping & Retail",
    "Convenience Store": "Shopping & Retail",
    "Cosmetics Shop": "Shopping & Retail",
    "Cupcake Shop": "Shopping & Retail",
    "Department Store": "Shopping & Retail",
    "Electronics Store": "Shopping & Retail",
    "Flea Market": "Shopping & Retail",
    "Flower Shop": "Shopping & Retail",
    "Furniture / Home Store": "Shopping & Retail",
    "Garden Center": "Shopping & Retail",
    "Gift Shop": "Shopping & Retail",
    "Hardware Store": "Shopping & Retail",
    "Hobby Shop": "Shopping & Retail",
    "Jewelry Store": "Shopping & Retail",
    "Mall": "Shopping & Retail",
    "Market": "Shopping & Retail",
    "Miscellaneous Shop": "Shopping & Retail",
    "Mobile Phone Shop": "Shopping & Retail",
    "Motorcycle Shop": "Shopping & Retail",
    "Music Store": "Shopping & Retail",
    "Newsstand": "Shopping & Retail",
    "Paper / Office Supplies Store": "Shopping & Retail",
    "Pet Store": "Shopping & Retail",
    "Record Shop": "Shopping & Retail",
    "Shop & Service": "Shopping & Retail",
    "Smoke Shop": "Shopping & Retail",
    "Sporting Goods Shop": "Shopping & Retail",
    "Thrift / Vintage Store": "Shopping & Retail",
    "Toy / Game Store": "Shopping & Retail",
    "Video Game Store": "Shopping & Retail",
    "Video Store": "Shopping & Retail",
    # ── Arts & Culture ────────────────────────────────────────────────────
    "Art Gallery": "Arts & Culture",
    "Art Museum": "Arts & Culture",
    "Concert Hall": "Arts & Culture",
    "Historic Site": "Arts & Culture",
    "History Museum": "Arts & Culture",
    "Museum": "Arts & Culture",
    "Performing Arts Venue": "Arts & Culture",
    "Planetarium": "Arts & Culture",
    "Public Art": "Arts & Culture",
    "Science Museum": "Arts & Culture",
    "Sculpture Garden": "Arts & Culture",
    "Theater": "Arts & Culture",
    # ── Entertainment ─────────────────────────────────────────────────────
    "Arts & Entertainment": "Entertainment",
    "Arcade": "Entertainment",
    "Fair": "Entertainment",
    "Gaming Cafe": "Entertainment",
    "General Entertainment": "Entertainment",
    "Internet Cafe": "Entertainment",
    "Movie Theater": "Entertainment",
    "Pool Hall": "Entertainment",
    # ── Outdoors & Recreation ─────────────────────────────────────────────
    "Aquarium": "Outdoors & Recreation",
    "Beach": "Outdoors & Recreation",
    "Campground": "Outdoors & Recreation",
    "Garden": "Outdoors & Recreation",
    "Harbor / Marina": "Outdoors & Recreation",
    "Other Great Outdoors": "Outdoors & Recreation",
    "Outdoors & Recreation": "Outdoors & Recreation",
    "Park": "Outdoors & Recreation",
    "Playground": "Outdoors & Recreation",
    "Plaza": "Outdoors & Recreation",
    "Pool": "Outdoors & Recreation",
    "River": "Outdoors & Recreation",
    "Scenic Lookout": "Outdoors & Recreation",
    "Zoo": "Outdoors & Recreation",
    # ── Sports & Fitness ──────────────────────────────────────────────────
    "Athletic & Sport": "Sports & Fitness",
    "Bowling Alley": "Sports & Fitness",
    "College Stadium": "Sports & Fitness",
    "Gym / Fitness Center": "Sports & Fitness",
    "Racetrack": "Sports & Fitness",
    "Stadium": "Sports & Fitness",
    # ── Education ─────────────────────────────────────────────────────────
    "College & University": "Education",
    "College Academic Building": "Education",
    "Community College": "Education",
    "Elementary School": "Education",
    "General College & University": "Education",
    "High School": "Education",
    "Law School": "Education",
    "Library": "Education",
    "Medical School": "Education",
    "Middle School": "Education",
    "Nursery School": "Education",
    "School": "Education",
    "Sorority House": "Education",
    "Student Center": "Education",
    "Trade School": "Education",
    "University": "Education",
    # ── Transportation ────────────────────────────────────────────────────
    "Airport": "Transportation",
    "Bike Rental / Bike Share": "Transportation",
    "Bus Station": "Transportation",
    "Ferry": "Transportation",
    "General Travel": "Transportation",
    "Light Rail": "Transportation",
    "Parking": "Transportation",
    "Rest Area": "Transportation",
    "Subway": "Transportation",
    "Taxi": "Transportation",
    "Train Station": "Transportation",
    "Travel & Transport": "Transportation",
    "Travel Lounge": "Transportation",
    # ── Services & Errands ────────────────────────────────────────────────
    "Animal Shelter": "Services & Errands",
    "Bank": "Services & Errands",
    "Drugstore / Pharmacy": "Services & Errands",
    "Financial or Legal Service": "Services & Errands",
    "Funeral Home": "Services & Errands",
    "Gas Station / Garage": "Services & Errands",
    "Laundry Service": "Services & Errands",
    "Medical Center": "Services & Errands",
    "Post Office": "Services & Errands",
    "Rental Car Location": "Services & Errands",
    "Recycling Facility": "Services & Errands",
    "Salon / Barbershop": "Services & Errands",
    "Spa / Massage": "Services & Errands",
    "Tanning Salon": "Services & Errands",
    "Tattoo Parlor": "Services & Errands",
    # ── Accommodation ─────────────────────────────────────────────────────
    "Hotel": "Accommodation",
    # ── Religious & Spiritual ─────────────────────────────────────────────
    "Church": "Religious & Spiritual",
    "Shrine": "Religious & Spiritual",
    "Spiritual Center": "Religious & Spiritual",
    "Synagogue": "Religious & Spiritual",
    "Temple": "Religious & Spiritual",
    # ── Work & Office ─────────────────────────────────────────────────────
    "Embassy / Consulate": "Work & Office",
    "Factory": "Work & Office",
    "Office": "Work & Office",
    "Government Building": "Work & Office",
    # ── Work & Event ─────────────────────────────────────────────────────
    "Design Studio": "Work & Event",
    "Convention Center": "Work & Event",
    "Event Space": "Work & Event",
    # ── Residential ───────────────────────────────────────────────────────
    "Home (private)": "Residential",
    "Housing Development": "Residential",
    "Residential Building (Apartment / Condo)": "Residential",
    # ── Neutral / Generic ─────────────────────────────────────────────────
    "City": "Neutral / Generic",
    "Bridge": "Neutral / Generic",
    "Building": "Neutral / Generic",
    "Cemetery": "Neutral / Generic",
    "Military Base": "Neutral / Generic",
    "Moving Target": "Neutral / Generic",
    "Neighborhood": "Neutral / Generic",
    "Professional & Other Places": "Neutral / Generic",
    "Road": "Neutral / Generic",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Level 2 → Level 3:  Mid-level  →  Segmentation macro
# ═══════════════════════════════════════════════════════════════════════════════

MID_TO_SEGMENTATION: dict[str, str] = {
    "Food & Dining": "FoodDrink",
    "Café & Coffee": "FoodDrink",
    "Bars & Nightlife": "NightlifeLeisure",
    "Shopping & Retail": "Shopping",
    "Arts & Culture": "Leisure",
    "Entertainment": "Leisure",
    "Outdoors & Recreation": "Leisure",
    "Sports & Fitness": "Leisure",
    "Education": "WorkEducation",
    "Transportation": "Transit",
    "Services & Errands": "ServiceErrand",
    "Accommodation": "StayResidential",
    "Religious & Spiritual": "CivicSpiritual",
    "Work & Office": "WorkEducation",
    "Work & Event": "WorkEvent",
    "Residential": "StayResidential",
    "Neutral / Generic": "Neutral",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: Raw → Segmentation in one step
# ═══════════════════════════════════════════════════════════════════════════════

RAW_TO_SEGMENTATION: dict[str, str] = {
    raw: MID_TO_SEGMENTATION[mid] for raw, mid in RAW_TO_MID.items()
}

# Sentinel labels
NEUTRAL_LABEL = "Neutral"
UNKNOWN_MID = "Unknown"
UNKNOWN_SEG = "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: Raw → Segmentation in one step
# ═══════════════════════════════════════════════════════════════════════════════

RAW_TO_SEGMENTATION: dict[str, str] = {
    raw: MID_TO_SEGMENTATION[mid] for raw, mid in RAW_TO_MID.items()
}

# Sentinel labels
NEUTRAL_LABEL = "Neutral"
TRANSIT_LABEL = "Transit"
TRANSIT_UNRESOLVED_LABEL = "Transit_Unresolved"
UNKNOWN_MID = "Unknown"
UNKNOWN_SEG = "Unknown"

# Labels that are absorbable (not standalone activity intents)
ABSORBABLE_LABELS = {NEUTRAL_LABEL, TRANSIT_LABEL}
