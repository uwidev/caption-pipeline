"""
ToriiGate prompt templates for different caption formats.

These prompts are used by TagNaturalLanguageStep to generate
natural language captions from images.
"""

from typing import Final

# Prompt templates for different caption types
TORIIGATE_PROMPTS: Final[dict[str, str]] = {
    "long_thoughts_v2": """Your answer must contain 6 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture with given popular tags, or descriptions, or your memories for each characters to determine who is who.
# 2. Key details
Here you need to determine key details on comic and list them.
# 3. Long description
Here come up with a long and detailed description of image content. Be creative, mention all detailes you listed above and other important things.
# 4. Detailed description for each character
## Name 1
Detailed and long description for the first character
## Name 2
Same for each one (if present)
</format>
""",

    "long_thoughts": """Your answer must contain 6 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture  with given popular tags, or descriptions, or your memories for each characters to determine who is who.
If no characters are listed in input - just write here "No named characters"
# 2. General description
A one-two paragraph summary of the image. Mention all individual parts/objects/characters/positions/interactions/etc.
# 3. Detailed description for each character
## Character name 1 (put here the name if any)
In very detail write about features, poses, look, used objects, interactions, and other things for character on the picture.
## Character name 2 (put here the name if any)
Same for each character.
...
# 4. Individual Parts
List the individual things you see in the image and their relative positions to other parts. Use a numbered list of between 5 and 20 items depending on image complexity.
# 5. Texts on image
Mention every texts that you notice on image, including types (a speech bubble, watermark, banner, etc.) and content.
# 6. Background and effects
Give some info about objects on background, describe the location (if seen). Then mention effects (style, camera angle, clarity/blurrines, effects like depth of field, strange angle/forshortening, etc.)
</format>
""",

    "json": """Use json-style caption for given image with following structure:
{"character" : "Description for character or object. Name (if defined), main details, features, position, pose, etc.",
/or in case of multiple
"character_1" : "Description for first"
"character_2" : "Description for second ",
"character_N"...
/or if there are no characters
"main content" : "long and detailed description of main content of image that might be the main focus if characters are missing",
/
"background" : "Detailed descritpion of background and it's content",
"image_effects" : "If there are some visual effects like fisheye distortion, chromatic aberration, glitches, messy drawing or anything else - write about it. If it's just a general anime art - omit this field."
"texts" : "Speech bubbles, bars, marks, signs etc. with texts if present, else None",
"atmosphere" : "...",
}
In special cases you can add extra keys.
""",

    "long": """Make a caption for given image with natural text. Use 2 to 5 paragraphs. Make your description long and vivid, mentioning all the details.
""",

    "min_structured_md": """Your answer must contain 3 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture  with given popular tags, or descriptions, or your memories for each characters to determine who is who.
If no characters are listed in input - just write here "No named characters"
# 2. Key details
Here you need to write about the key details on image, prefere using regular text.
# 3. Structured description
## General
Write about general composition, content of image, background and all things that are not related to characters directly.
## Character name 1 (put here the name if any)
Write about datails and content related to specific character, including features, poses, look, used objects, interactions, and other things.
## Character name 2 (put here the name if any)
Same for each character.
## Image effects
Mention image effect, style, camera angle
</format>
In general stick to shorter descriptions.
""",

    "json_comic": """Use json-style caption to describe to comin, stick to following structure:
{
"comic_format": "menation the format, for example Comic of N frames",
"1st_frame": "Main description of the content for fist frame",
"2nd_frame": "Same for the second",
...
"Nth_ftame": "...",
"character_1": "Describe the characters in comic",
...
"character_N": "Separate description for each",
"meaning": "Try to guess general mood, vibe and meaning of the comic"
}
""",

    "md_comic": """Use markdown format to describe to comic, 5 parts are recommended:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture with given popular tags, or descriptions, or your memories for each characters to determine who is who.
# 2. Key details
Here you need to determine key details on comic and list them.
# 3. Comic format
In this section come up with the description of comic format, how many pages there are, horisontal/vertical orientation and other things. Optionally you can list main characters here.
# 4. Details for each frame
## 4.1 Frame 1 (position)
Description for each frame, includding characters, objects, interactions, texts/speech bubbles and other things. Be detailed but not overdoo.
## 4.2 Frame 2 (position)
Same for each frame.
...
# 5. Extra comment
Here you should write general desciption and some other info about the image.
</format>
""",

    "min_structured_json": """
Use json-style caption for given image with following structure:
{"General" : "Here you need to come up with general/common information about picture, overall composition. Stick to shorter phrases and tags instead of long purple prose. Avoid bullets and markdown, write in plain text.",
"character_1 (put here the name if any)" : "Description of first character."
"character_2 (if present" : "Description for second ",
"character_N"
...
"image_effects" : "Mention here effects on image if there are any distinct."
"texts" : "Speech bubbles, bars, marks, signs etc. with texts if present, else None",
"watermarks" : "If present",
}
Prefere shorter description and tags.
""",

    "chroma-style": """Your task is to describe the picture in very detail using a structure of 4 parts.
### 1. Regular Summary:
[A one-paragraph summary of the image. The paragraph should mention all individual parts/things/characters/etc.]
### 2. Individual Parts:
[List the individual things you see in the image and their relative positions to other parts. Use a numbered list of between 5 and 30 items depending on image complexity.]
### 3. Midjourney-Style Summary:
[A summary that has higher concept density by using comma-separated partial sentences instead of proper sentence structure.]
### 4. DeviantArt Commission Request
[Write a description as if you're commissioning this *exact* image via someone who is currently taking requests.]
""",

    "short": """The caption for image should be quite short without long purple prose and slop. Cover main objects and details.
""",
}
