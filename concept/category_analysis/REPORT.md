# Concept counts by prompt category

Counts are images containing a detector label, not numbers of boxes. Repeated labels in an image count once. Targets are unreviewed predictions; missing labels may be detector misses. Clothing and hairstyle presence is not a racism label.

This pilot has 12 categories, 72 images total. The source dataset defines six race prompts and two gender prompts (12 categories), not 16. Each category currently has six images: three discovery and three validation. Test remains reserved.

| Category | Images | head_covering | turban | headscarf | earrings | necklace | beard | afro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| black man | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| white man | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| east asian man | 6 | 2 | 0 | 0 | 0 | 0 | 1 | 0 |
| middle eastern man | 6 | 5 | 1 | 3 | 0 | 0 | 6 | 0 |
| indian man | 6 | 5 | 5 | 0 | 0 | 2 | 5 | 0 |
| hispanic man | 6 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| black woman | 6 | 0 | 0 | 0 | 4 | 0 | 0 | 5 |
| white woman | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| east asian woman | 6 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| middle eastern woman | 6 | 3 | 0 | 2 | 1 | 0 | 0 | 0 |
| indian woman | 6 | 1 | 1 | 0 | 4 | 5 | 0 | 0 |
| hispanic woman | 6 | 1 | 0 | 0 | 6 | 6 | 0 | 0 |

| Category | Images | kimono | shirt | tank_top | smile | flower | red_dot |
|---|---:|---:|---:|---:|---:|---:|---:|
| black man | 6 | 0 | 3 | 0 | 3 | 0 | 0 |
| white man | 6 | 0 | 6 | 0 | 0 | 0 | 0 |
| east asian man | 6 | 0 | 3 | 0 | 2 | 0 | 0 |
| middle eastern man | 6 | 0 | 1 | 0 | 0 | 0 | 0 |
| indian man | 6 | 0 | 2 | 1 | 0 | 0 | 1 |
| hispanic man | 6 | 0 | 5 | 1 | 2 | 0 | 0 |
| black woman | 6 | 0 | 0 | 3 | 0 | 0 | 0 |
| white woman | 6 | 0 | 5 | 1 | 2 | 0 | 0 |
| east asian woman | 6 | 6 | 0 | 0 | 0 | 6 | 0 |
| middle eastern woman | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| indian woman | 6 | 0 | 0 | 0 | 0 | 0 | 4 |
| hispanic woman | 6 | 0 | 1 | 0 | 0 | 0 | 0 |

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

Category JPG sheets below show all unique phrase-grounding and object-detection boxes for each image. Numbered boxes correspond to the legend below each tile. Blue: phrase grounding; red: object detection. Both sources can disagree or miss concepts. Some large boxes label whole people or abstract descriptions rather than distinct objects. These boxes are not SAE heatmaps.

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
