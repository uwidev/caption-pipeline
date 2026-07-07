"""
Pre-categorization patterns for tag ordering.

This module defines flexible regex patterns that categorize tags into semantic groups
before falling back to LLM-based classification. The patterns are designed to match
Danbooru-style compound tags (suffix/prefix patterns only – no exact matches).

These patterns are used by FixOrderStep to reduce LLM calls for common tags.
"""

import re
from typing import Final

# Each entry: (compiled_regex, category)
# Order matters: more specific patterns should come first.
PRE_CATEGORIZE_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    # ---- Actions (prefix verb_object) ----
    # Catches: holding_sword, lifting_skirt, looking_at_viewer, etc.
    (re.compile(r"^(holding|wearing|removing|taking_off|putting_on|lifting|pulling|pushing|carrying|dragging|riding|driving|flying|swimming|diving|jumping|leaping|climbing|kicking|punching|grabbing|throwing|catching|pointing_at|reaching_for|looking_at|looking_to|staring_at|glaring_at|waving_at|talking_to|shouting_at|whispering_to|kissing|hugging|cuddling|patting|stroking|groping|spanking|biting|licking|sucking|shooting|stabbing|slashing|cutting|breaking|picking_up|putting_down|opening|closing|washing|cleaning|cooking|eating|drinking|playing|singing|dancing)_"), "action"),

    # ---- Actions (gaze) ----
    (re.compile(r"^looking.*"), "action"),

    # ---- Pose (prefix: on_bed, on_floor, on_table, etc.) ----
    (re.compile(r"^on_.*"), "pose"),

    # ---- Pose (suffix: body arrangements) ----
    # Catches: arms_crossed, legs_spread, hands_up, etc.
    (re.compile(r".*_(crossed|spread|raised|tilted|up|down|out|back)$"), "pose"),

    # ---- Body parts: any tag ending with common body part names ----
    (re.compile(r".*_(hair|eyes|tail|ears|horns|wings|skin|fur|claws|hands|feet|legs|arms|breasts|chest|back|neck|head|face|mouth|nose|chin|thighs|calves|toes|fingers|paws|hooves|antlers|tentacles|scales|feathers|eye)$"), "body parts"),

    # ---- Wearables: any tag ending with clothing/accessory terms ----
    (re.compile(r".*_(shirt|dress|skirt|pants|jacket|coat|hat|glasses|ribbon|bow|socks|shoes|gloves|belt|scarf|tie|necklace|earrings|bracelet|watch|vest|sweater|hoodie|bikini|swimsuit|underwear|panties|bra|shorts|skort|blouse|trousers|jeans|sweatpants|leggings|stockings|garter|suspenders|apron|cape|cloak|hood|mask|headphones|crown|tiara|flower|armband|wristband|hairband|hairclip|hairpin)$"), "wearables"),

    # ---- Environment: productive suffixes ----
    (re.compile(r".*_(background|sky|clouds|room|ocean|city|sunset|sunrise)$"), "environment"),

    # ---- Atmosphere: lighting-related suffixes ----
    (re.compile(r".*_(lighting|light|shade|shadow|tone)$"), "atmosphere"),
]
