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

## Quick Start

To evaluate text-to-image models on our T2I-CoReBench, follow these steps:

### Generate Images

Use the provided script to generate images from the benchmark prompts in `./data`. You can customize the T2I models by editing `MODELS` and adjust GPU usage by setting `GPUS`. Here, we take *Qwen-Image* as an example, and the corresponding Python environment can be referred to in its [official repository](https://github.com/QwenLM/Qwen-Image).

  ```bash
  bash sample.sh
  ```

If you wish to sample with your own model, simply modify the sampling code in `sample.py`, i.e., the model loading part (lines 44–72) and the sampling part (line 94); no other changes are required.

### Run Evaluation

Evaluate the generated images using our evaluation framework. We provide evaluation code based on both **Gemini 2.5 Flash** and **Qwen2.5-VL-72B**. For environment setup, please refer to the [Gemini documentation](https://ai.google.dev/gemini-api/docs) (an official API key is required) and the [vLLM User Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#installation). When using **Qwen2.5-VL-72B** for evaluation, our experiments are conducted with 8 × A800 GPUs (80GB); however, our tests confirm that it can also run with 4 × A800 GPUs.

  ```bash
  bash eval.sh
  ```

The evaluation process will automatically assess the generated images across all 12 dimensions of our benchmark and provide a `mean_score` for each dimension in an individual `json` file.


## Examples of Each Dimension

<p align="center">
  <img src="assets/fig_composition.jpeg" width="95%"><br>
  <em>(a) Composition (i.e., MI, MA, MR, TR)</em>
</p>

<p align="center">
  <img src="assets/fig_reasoning_de.jpeg" width="95%"><br>
  <em>(b) Deductive Reasoning (i.e., LR, BR, HR, PR)</em>
</p>

<p align="center">
  <img src="assets/fig_reasoning_in.jpeg" width="95%"><br>
  <em>(c) Inductive Reasoning (i.e., GR, AR)</em>
</p>

<p align="center">
  <img src="assets/fig_reasoning_ab.jpeg" width="95%"><br>
  <em>(d) Abductive Reasoning (i.e., CR, RR)</em>
</p>

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