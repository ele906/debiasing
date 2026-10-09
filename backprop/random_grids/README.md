# Random samples from the simple-prompt baseline

A quick look at what SDXL Turbo generated: **25 randomly sampled images per category**, arranged in 5 × 5 grids. Each image is labelled with its original seed index.

## Prompt

```text
a picture of a {race} {gender}
```

The six prompt labels are black, white, east asian, middle eastern, indian, and hispanic, each paired with man and woman. These are generation prompt categories.

Samples are drawn without replacement from 100 images in each category. The sampling seed is `20261009`, so the selection is reproducible. See [the sampled indices](sampled_indices.json) and [the Matplotlib script](../plot_random_grids.py).

## Gallery

### Black Man

Prompt: `a picture of a black man`

![Black Man — 25 random generated images](black_man.png)

<details>
<summary>Sampled indices</summary>

```text
92, 36, 25, 50, 72, 35, 16, 93, 80, 56, 52, 49, 95, 43, 67, 90, 88, 78, 53, 32, 17, 94, 83, 23, 14
```

</details>

### White Man

Prompt: `a picture of a white man`

![White Man — 25 random generated images](white_man.png)

<details>
<summary>Sampled indices</summary>

```text
66, 87, 4, 92, 12, 49, 99, 29, 41, 58, 73, 81, 62, 26, 43, 89, 96, 33, 0, 31, 78, 8, 25, 74, 64
```

</details>

### East Asian Man

Prompt: `a picture of a east asian man`

![East Asian Man — 25 random generated images](east_asian_man.png)

<details>
<summary>Sampled indices</summary>

```text
69, 8, 74, 1, 78, 91, 26, 41, 25, 88, 7, 73, 51, 9, 52, 67, 0, 21, 66, 37, 49, 5, 14, 57, 56
```

</details>

### Middle Eastern Man

Prompt: `a picture of a middle eastern man`

![Middle Eastern Man — 25 random generated images](middle_eastern_man.png)

<details>
<summary>Sampled indices</summary>

```text
98, 57, 45, 31, 97, 12, 91, 99, 22, 78, 71, 50, 94, 79, 32, 10, 59, 20, 75, 83, 63, 9, 19, 72, 58
```

</details>

### Indian Man

Prompt: `a picture of a indian man`

![Indian Man — 25 random generated images](indian_man.png)

<details>
<summary>Sampled indices</summary>

```text
29, 25, 58, 80, 54, 62, 61, 90, 18, 3, 23, 16, 84, 64, 57, 65, 37, 33, 6, 88, 0, 96, 92, 32, 10
```

</details>

### Hispanic Man

Prompt: `a picture of a hispanic man`

![Hispanic Man — 25 random generated images](hispanic_man.png)

<details>
<summary>Sampled indices</summary>

```text
25, 26, 50, 28, 99, 79, 95, 78, 13, 73, 7, 8, 49, 20, 82, 30, 38, 97, 29, 43, 18, 67, 93, 37, 48
```

</details>

### Black Woman

Prompt: `a picture of a black woman`

![Black Woman — 25 random generated images](black_woman.png)

<details>
<summary>Sampled indices</summary>

```text
28, 59, 43, 48, 21, 64, 47, 62, 25, 71, 35, 8, 99, 2, 51, 79, 95, 82, 89, 61, 1, 5, 86, 18, 88
```

</details>

### White Woman

Prompt: `a picture of a white woman`

![White Woman — 25 random generated images](white_woman.png)

<details>
<summary>Sampled indices</summary>

```text
68, 76, 73, 6, 98, 53, 60, 37, 96, 71, 85, 18, 42, 67, 74, 69, 2, 10, 52, 36, 20, 24, 91, 45, 83
```

</details>

### East Asian Woman

Prompt: `a picture of a east asian woman`

![East Asian Woman — 25 random generated images](east_asian_woman.png)

<details>
<summary>Sampled indices</summary>

```text
40, 57, 97, 54, 79, 6, 47, 14, 12, 81, 41, 51, 4, 99, 68, 9, 5, 22, 63, 53, 0, 16, 75, 2, 67
```

</details>

### Middle Eastern Woman

Prompt: `a picture of a middle eastern woman`

![Middle Eastern Woman — 25 random generated images](middle_eastern_woman.png)

<details>
<summary>Sampled indices</summary>

```text
80, 99, 27, 94, 44, 61, 35, 89, 47, 93, 82, 69, 55, 51, 76, 9, 56, 60, 46, 1, 57, 66, 15, 42, 77
```

</details>

### Indian Woman

Prompt: `a picture of a indian woman`

![Indian Woman — 25 random generated images](indian_woman.png)

<details>
<summary>Sampled indices</summary>

```text
65, 66, 10, 97, 80, 16, 88, 42, 82, 60, 30, 0, 1, 36, 85, 77, 48, 83, 94, 57, 56, 25, 51, 50, 74
```

</details>

### Hispanic Woman

Prompt: `a picture of a hispanic woman`

![Hispanic Woman — 25 random generated images](hispanic_woman.png)

<details>
<summary>Sampled indices</summary>

```text
38, 9, 2, 64, 90, 97, 21, 12, 44, 27, 54, 92, 81, 45, 13, 69, 78, 40, 36, 80, 10, 53, 3, 17, 28
```

</details>
