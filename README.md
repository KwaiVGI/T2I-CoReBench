# T2I-CoReBench

This is the official Pytorch implementation of our paper:

> [Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?](https://arxiv.org/abs/2509.03516)
>
> Ouxiang Li, Yuan Wang, Xinting Hu, Huijuan Huang, Rui Chen, Jiarong Ou, Xin Tao, Pengfei Wan, Fuli Feng

<p align="center">
  <a href='https://t2i-corebench.github.io/'>
    <img src='https://img.shields.io/badge/Project Page-0065D3?logo=rocket&logoColor=white'>
  </a>
  <a href='https://arxiv.org/abs/2509.03516'>
    <img src='https://img.shields.io/badge/Arxiv-2509.03516-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
</p>


![teaser](assets/teaser.jpeg)

**Statistics of T2I-CoReBench.** *Left*: Our T2I evaluation taxonomy spanning two fundamental generative capabilities (i.e., *composition* and *reasoning*), further refined into 12 dimensions. *Right*: Distributions of prompt-token lengths and checklist-question counts. Our benchmark demonstrates high complexity, with an average prompt length of 170 tokens and an average of 12.5 questions. Note: reasoning has fewer questions, as each requires reasoning that is more challenging.

## News
- `2025/09` :star2: We have released our benchmark dataset and code.

## Benchmark Comparison

![benchmark_comparison](assets/benchmark_comparison.jpeg)

T2I-CoReBench comprehensively covers 12 evaluation dimensions spanning both *composition* and *reasoning* scenarios. The symbols indicate different coverage levels: <span style="font-size:32px; vertical-align: -5px; line-height:1;">●</span> means coverage with high compositional (visual elements > 5) or reasoning (one-to-many or many-to-one inference) complexity. <span style="font-size:16px; line-height:1;">◐</span> means coverage under simple settings (visual elements ≤ 5 or one-to-one inference). <span style="font-size:32px; vertical-align: -5px; line-height:1;">○</span> means this dimension is not covered.

## Evaluation Guideline


## Examples of Each Dimension

<figure style="text-align:center; margin:1.5em 0;">
  <img src="assets/fig_composition.jpeg" style="max-width:95%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
  <figcaption style="margin-top:0.5em; font-size:14px; color:#555;">
    (a) Composition (i.e., MI, MA, MR, TR)
  </figcaption>
</figure>

<figure style="text-align:center; margin:1.5em 0;">
  <img src="assets/fig_reasoning_de.jpeg" style="max-width:95%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
  <figcaption style="margin-top:0.5em; font-size:14px; color:#555;">
    (b) Deductive Reasoning (i.e., LR, BR, HR, PR)
  </figcaption>
</figure>

<figure style="text-align:center; margin:1.5em 0;">
  <img src="assets/fig_reasoning_in.jpeg" style="max-width:95%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
  <figcaption style="margin-top:0.5em; font-size:14px; color:#555;">
    (c) Inductive Reasoning (i.e., GR, AR)
  </figcaption>
</figure>

<figure style="text-align:center; margin:1.5em 0;">
  <img src="assets/fig_reasoning_ab.jpeg" style="max-width:95%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
  <figcaption style="margin-top:0.5em; font-size:14px; color:#555;">
    (d) Abductive Reasoning (i.e., CR, RR)
  </figcaption>
</figure>

## Citation
If you find the repo useful, please consider citing.
```
@misc{li2025easier,
      title={Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?}, 
      author={Ouxiang Li and Yuan Wang and Xinting Hu and Huijuan Huang and Rui Chen and Jiarong Ou and Xin Tao and Pengfei Wan and Fuli Feng},
      year={2025},
      eprint={2509.03516},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.03516}, 
}
```