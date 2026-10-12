# Concept counts by prompt category

Counts are images containing a detector label, not numbers of boxes. Repeated labels in an image count once. Targets are unreviewed predictions; missing labels may be detector misses. Clothing and hairstyle presence is not a racism label.

This extraction has 12 categories, 1200 images total. The source dataset defines six race prompts and two gender prompts (12 categories), not 16. Counts use every supplied image; denominators appear in the table. Test labels, when supplied, are descriptive only and must be excluded from SAE feature selection.

| Category | Images | head_covering | turban | headscarf | earrings | necklace | beard | afro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| black man | 100 | 0 | 0 | 0 | 0 | 0 | 6 | 2 |
| white man | 100 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| east asian man | 100 | 22 | 0 | 0 | 0 | 0 | 22 | 0 |
| middle eastern man | 100 | 84 | 27 | 40 | 0 | 3 | 100 | 0 |
| indian man | 100 | 93 | 93 | 0 | 3 | 45 | 89 | 0 |
| hispanic man | 100 | 5 | 0 | 0 | 1 | 5 | 78 | 0 |
| black woman | 100 | 1 | 0 | 0 | 91 | 21 | 0 | 79 |
| white woman | 100 | 0 | 0 | 0 | 2 | 1 | 0 | 0 |
| east asian woman | 100 | 7 | 0 | 0 | 34 | 0 | 0 | 0 |
| middle eastern woman | 100 | 46 | 0 | 34 | 29 | 0 | 0 | 0 |
| indian woman | 100 | 3 | 1 | 1 | 75 | 93 | 0 | 0 |
| hispanic woman | 100 | 14 | 0 | 1 | 85 | 90 | 0 | 0 |

| Category | Images | kimono | shirt | tank_top | smile | flower | red_dot |
|---|---:|---:|---:|---:|---:|---:|---:|
| black man | 100 | 0 | 67 | 0 | 44 | 0 | 0 |
| white man | 100 | 0 | 95 | 0 | 11 | 0 | 0 |
| east asian man | 100 | 17 | 51 | 0 | 34 | 2 | 0 |
| middle eastern man | 100 | 0 | 27 | 0 | 0 | 0 | 0 |
| indian man | 100 | 0 | 28 | 1 | 0 | 0 | 7 |
| hispanic man | 100 | 0 | 81 | 12 | 17 | 0 | 0 |
| black woman | 100 | 0 | 0 | 24 | 3 | 1 | 0 |
| white woman | 100 | 0 | 61 | 5 | 18 | 0 | 0 |
| east asian woman | 100 | 96 | 10 | 0 | 2 | 59 | 0 |
| middle eastern woman | 100 | 0 | 2 | 0 | 5 | 0 | 2 |
| indian woman | 100 | 0 | 1 | 0 | 0 | 0 | 51 |
| hispanic woman | 100 | 0 | 29 | 1 | 0 | 6 | 0 |

## How labels are matched

The normalized table uses these exact detector-phrase patterns (caption-only mentions are excluded):

- `head_covering`: `\b(head\s*scarf|head\s*covering|headdress|turban|hat|headband)\b`
- `turban`: `\bturban\b`
- `headscarf`: `\b(head\s*scarf|headdress)\b`
- `earrings`: `\bearrings?\b`
- `necklace`: `\bnecklace\b`
- `beard`: `\b(beard|goatee)\b`
- `afro`: `\bafro\b`
- `kimono`: `\bkimono\b`
- `shirt`: `\bshirt\b`
- `tank_top`: `\btank top\b`
- `smile`: `\bsmile\b`
- `flower`: `\bflowers?\b`
- `red_dot`: `\bred dot\b`

Raw phrases are preserved in raw_phrase_counts_by_category.csv. Rates are fractions from 0 to 1 in rates_by_category.csv. The pilot is small and these are descriptive counts, not population estimates or significance tests.

## Bounding boxes

Category JPG sheets below show all unique phrase-grounding and object-detection boxes for each image. Numbered boxes correspond to the legend below each tile. Blue: phrase grounding; red: object detection. Both sources can disagree or miss concepts. Malformed reversed-coordinate boxes are preserved in the raw JSON but omitted from these drawings. Some large boxes label whole people or abstract descriptions rather than distinct objects. These boxes are not SAE heatmaps.

Category sheets show the first 6 examples per category. Individual boxed images exist under boxes/ for every supplied sample.

### black man

![black_man](black_man_boxes.jpg)

### white man

![white_man](white_man_boxes.jpg)

### east asian man

![east_asian_man](east_asian_man_boxes.jpg)

### middle eastern man

![middle_eastern_man](middle_eastern_man_boxes.jpg)

### indian man

![indian_man](indian_man_boxes.jpg)

### hispanic man

![hispanic_man](hispanic_man_boxes.jpg)

### black woman

![black_woman](black_woman_boxes.jpg)

### white woman

![white_woman](white_woman_boxes.jpg)

### east asian woman

![east_asian_woman](east_asian_woman_boxes.jpg)

### middle eastern woman

![middle_eastern_woman](middle_eastern_woman_boxes.jpg)

### indian woman

![indian_woman](indian_woman_boxes.jpg)

### hispanic woman

![hispanic_woman](hispanic_woman_boxes.jpg)
