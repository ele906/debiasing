# Top discovery features by layer and prompt group

This table lists the top 3 features selected using **only discovery seeds 100–149**.
The later signed image sweeps use held-out seed(s), so held-out images do not choose these IDs.

`contrast` is the mean feature activation in the target prompt group minus the
same-seed average over the other five race prompts of the same gender, within the
face box. It is a raw SAE-activation difference, not a semantic label, fairness
score, or image-quality score.

## down.2.1 (`unet.down_blocks.2.attentions.1`)

| Prompt group | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| black man | F527 (+27.296) | F4022 (+4.206) | F2137 (+3.579) |
| east asian man | F3722 (+8.930) | F1759 (+7.400) | F1600 (+6.109) |
| hispanic man | F1653 (+6.760) | F199 (+3.676) | F1678 (+2.439) |
| indian man | F1050 (+17.097) | F3722 (+12.901) | F669 (+6.080) |
| middle eastern man | F138 (+13.337) | F452 (+12.500) | F669 (+9.743) |
| white man | F2707 (+12.310) | F4262 (+8.968) | F243 (+8.289) |
| black woman | F527 (+37.929) | F243 (+10.079) | F3279 (+8.430) |
| east asian woman | F1759 (+19.653) | F2810 (+14.564) | F1600 (+8.004) |
| hispanic woman | F527 (+5.515) | F4022 (+3.698) | F1584 (+3.331) |
| indian woman | F3725 (+24.864) | F3722 (+14.975) | F4858 (+9.361) |
| middle eastern woman | F138 (+31.088) | F3462 (+7.217) | F3701 (+6.752) |
| white woman | F4384 (+10.895) | F472 (+6.630) | F2250 (+6.386) |

## mid.0 (`unet.mid_block.attentions.0`)

| Prompt group | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| black man | F2138 (+3.418) | F2875 (+1.257) | F4338 (+1.230) |
| east asian man | F2385 (+2.439) | F1424 (+2.130) | F3134 (+1.769) |
| hispanic man | F4601 (+1.376) | F826 (+0.856) | F169 (+0.832) |
| indian man | F4228 (+3.259) | F2023 (+2.105) | F3770 (+1.848) |
| middle eastern man | F4228 (+7.196) | F2023 (+5.249) | F3770 (+4.528) |
| white man | F4338 (+6.220) | F3482 (+2.771) | F1874 (+1.339) |
| black woman | F4338 (+4.370) | F1650 (+2.835) | F2138 (+1.801) |
| east asian woman | F1424 (+2.300) | F3828 (+2.267) | F846 (+2.050) |
| hispanic woman | F1424 (+2.361) | F3828 (+2.289) | F3770 (+2.114) |
| indian woman | F3018 (+2.480) | F1874 (+2.438) | F2023 (+2.183) |
| middle eastern woman | F2023 (+4.046) | F5029 (+3.374) | F4399 (+2.452) |
| white woman | F2361 (+8.056) | F5094 (+6.557) | F4786 (+2.895) |

## up.0.0 (`unet.up_blocks.0.attentions.0`)

| Prompt group | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| black man | F2975 (+2.865) | F1573 (+2.059) | F3111 (+1.967) |
| east asian man | F641 (+2.573) | F2854 (+2.554) | F573 (+2.263) |
| hispanic man | F1573 (+2.591) | F3339 (+2.107) | F3137 (+1.491) |
| indian man | F2261 (+9.241) | F4540 (+5.154) | F3742 (+4.570) |
| middle eastern man | F3742 (+11.348) | F4540 (+7.621) | F2261 (+7.016) |
| white man | F2975 (+4.915) | F1811 (+3.535) | F1354 (+3.224) |
| black woman | F4782 (+5.711) | F4163 (+3.445) | F2975 (+2.516) |
| east asian woman | F4540 (+5.425) | F4323 (+3.719) | F4367 (+3.081) |
| hispanic woman | F2706 (+2.484) | F4621 (+2.135) | F641 (+1.618) |
| indian woman | F4099 (+11.057) | F4540 (+7.280) | F2706 (+4.252) |
| middle eastern woman | F4099 (+13.546) | F1088 (+10.969) | F4540 (+5.594) |
| white woman | F4376 (+8.971) | F595 (+6.045) | F1343 (+3.945) |

## up.0.1 (`unet.up_blocks.0.attentions.1`)

| Prompt group | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| black man | F4464 (+12.334) | F4408 (+10.772) | F4044 (+9.710) |
| east asian man | F1042 (+12.094) | F383 (+6.748) | F1130 (+6.188) |
| hispanic man | F76 (+6.127) | F4922 (+5.877) | F3936 (+4.965) |
| indian man | F2784 (+16.400) | F1243 (+12.241) | F2913 (+7.569) |
| middle eastern man | F4147 (+11.188) | F1243 (+5.382) | F2355 (+3.285) |
| white man | F383 (+13.572) | F3166 (+10.219) | F2138 (+7.265) |
| black woman | F4408 (+19.481) | F383 (+10.809) | F4464 (+10.309) |
| east asian woman | F1042 (+20.962) | F2138 (+15.846) | F354 (+5.977) |
| hispanic woman | F3166 (+14.612) | F3936 (+4.200) | F500 (+4.141) |
| indian woman | F1243 (+21.367) | F3503 (+12.755) | F3507 (+6.859) |
| middle eastern woman | F1243 (+5.079) | F1779 (+4.211) | F4743 (+3.604) |
| white woman | F4461 (+8.634) | F2727 (+6.686) | F3132 (+5.689) |

## How to use the galleries

Each feature has an `activation_jet.jpg`, `signed_sweep.jpg`, and
`signed_coefficient_jet.jpg`. The sweep uses multipliers −5, −3, −2, −1,
0, 1, 2, 3, 5. `s = 1` is unchanged; `s = 0` removes the original feature
contribution; negative values reverse it. JET color runs blue → cyan → green →
yellow → orange → red. Read `REPORT.md` for the intervention equation and
interpretation limits.
