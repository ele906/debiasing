# Summary

Complete table from [summary.csv](summary.csv); 13 rows. No rows are omitted.

| concept | train_positive | train_negative | validation_positive | validation_negative | status | selection | feature | train_auc | validation_auc | train_matched_groups | validation_matched_groups | validation_within_difference | train_matched_auc | validation_matched_auc | train_positive_active | validation_positive_active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head_covering | 166 | 554 | 51 | 189 | within-group controls available; inspect validation effect | within_group_auc | 4154 | 0.4808620764647036 | 0.44926859632741983 | 7 | 3 | 0.02571294030381575 | 0.6621753572052871 | 0.6267045454545455 | 162 | 50 |
| turban | 73 | 647 | 24 | 216 | within-group controls available; inspect validation effect | within_group_auc | 1526 | 0.617729880798628 | 0.6186342592592593 | 2 | 1 | -0.10410255779113087 | 0.7457792207792208 | 0.35714285714285715 | 41 | 14 |
| headscarf | 45 | 675 | 15 | 225 | within-group controls available; inspect validation effect | within_group_auc | 1257 | 0.3154567901234568 | 0.22755555555555557 | 2 | 2 | -0.019209150789517085 | 0.6181093681093681 | 0.40470085470085476 | 45 | 15 |
| earrings | 185 | 535 | 67 | 173 | within-group controls available; inspect validation effect | within_group_auc | 4858 | 0.4596110128820409 | 0.4072124924510396 | 5 | 3 | 0.01813353059565485 | 0.651895005516421 | 0.49526198439241914 | 185 | 67 |
| necklace | 155 | 565 | 52 | 188 | within-group controls available; inspect validation effect | within_group_auc | 3985 | 0.8018726805595204 | 0.8045212765957447 | 5 | 1 | 0.15293343406584528 | 0.6962790719632825 | 0.7424242424242424 | 145 | 48 |
| beard | 179 | 541 | 60 | 180 | within-group controls available; inspect validation effect | within_group_auc | 5085 | 0.9092307851175663 | 0.9102777777777777 | 4 | 2 | -0.01882251480128616 | 0.6670741601127556 | 0.421875 | 166 | 56 |
| afro | 46 | 674 | 17 | 223 | within-group controls available; inspect validation effect | within_group_auc | 1500 | 0.9578764030447684 | 0.9662358216829333 | 1 | 1 | 0.1538252912315668 | 0.72 | 0.6666666666666666 | 46 | 17 |
| kimono | 70 | 650 | 22 | 218 | insufficient within-group validation controls | within_group_auc | 5046 | 0.6404175824175824 | 0.627814845704754 | 2 | 0 |  | 0.7645810003732736 |  | 70 | 22 |
| shirt | 273 | 447 | 89 | 151 | within-group controls available; inspect validation effect | within_group_auc | 2858 | 0.6513263023330137 | 0.6154475779447875 | 8 | 7 | 0.009961654906453307 | 0.5613017017596913 | 0.5271888485123779 | 117 | 32 |
| tank_top | 22 | 698 | 14 | 226 | within-group controls available; inspect validation effect | within_group_auc | 318 | 0.5970304766866371 | 0.45116940581542353 | 3 | 2 | 0.009003690050486575 | 0.7341617933723198 | 0.49977718360071305 | 22 | 13 |
| smile | 71 | 649 | 32 | 208 | within-group controls available; inspect validation effect | within_group_auc | 3593 | 0.6542025651598342 | 0.6172626201923077 | 6 | 4 | 0.22720567992659824 | 0.6694225006463215 | 0.595948717948718 | 70 | 31 |
| flower | 40 | 680 | 15 | 225 | within-group controls available; inspect validation effect | within_group_auc | 2230 | 0.8594117647058823 | 0.8586666666666667 | 2 | 1 | -0.008498526754833424 | 0.6600378787878788 | 0.5476190476190477 | 40 | 15 |
| red_dot | 34 | 686 | 10 | 230 | within-group controls available; inspect validation effect | within_group_auc | 1817 | 0.43768221574344024 | 0.45956521739130435 | 2 | 1 | 0.05041539409395418 | 0.657097288676236 | 0.6363636363636364 | 33 | 10 |
