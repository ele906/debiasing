# 60-image SAE review packet

Five randomly selected portraits per race/gender prompt group: four train and one validation image per group. The held-out test split is untouched. Selection did not use concept labels or SAE feature values.

Open each group contact sheet, then record labels in `annotations.csv`: `1` = visible, `0` = absent, `uncertain` = cannot tell, blank = not reviewed. Label the visible object only; flower prints in clothing/background do not count as a flower object. Count a clearly visible kimono garment, and a clearly visible turban headwrap; do not infer an item from the person’s prompt category. Clothing and accessories are presence labels, not harm labels.

This is a small pilot. A concept may have too few positive examples, especially in one split. If so, do not train a probe from it; collect more reviewed examples.

## black man

![black man sample contact sheet](contact_sheets/black_man.jpg)

## black woman

![black woman sample contact sheet](contact_sheets/black_woman.jpg)

## white man

![white man sample contact sheet](contact_sheets/white_man.jpg)

## white woman

![white woman sample contact sheet](contact_sheets/white_woman.jpg)

## east asian man

![east asian man sample contact sheet](contact_sheets/east_asian_man.jpg)

## east asian woman

![east asian woman sample contact sheet](contact_sheets/east_asian_woman.jpg)

## middle eastern man

![middle eastern man sample contact sheet](contact_sheets/middle_eastern_man.jpg)

## middle eastern woman

![middle eastern woman sample contact sheet](contact_sheets/middle_eastern_woman.jpg)

## indian man

![indian man sample contact sheet](contact_sheets/indian_man.jpg)

## indian woman

![indian woman sample contact sheet](contact_sheets/indian_woman.jpg)

## hispanic man

![hispanic man sample contact sheet](contact_sheets/hispanic_man.jpg)

## hispanic woman

![hispanic woman sample contact sheet](contact_sheets/hispanic_woman.jpg)
