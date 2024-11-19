---
license: cc-by-4.0
title: ZIM demo
emoji: 📈
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 4.37.1
app_file: app.py
pinned: false
python_version: 3.10.12
short_description: 'ZIM: Zero-Shot Image Matting for Anything demo'
---

# ZIM: Zero-Shot Image Matting for Anything

## Introduction

🚀 Introducing ZIM: Zero-Shot Image Matting – A Step Beyond SAM! 🚀

While SAM (Segment Anything Model) has redefined zero-shot segmentation with broad applications across multiple fields, it often falls short in delivering high-precision, fine-grained masks. That’s where ZIM comes in.

🌟 What is ZIM? 🌟

ZIM (Zero-Shot Image Matting) is a groundbreaking model developed to set a new standard in precision matting while maintaining strong zero-shot capabilities. Like SAM, ZIM can generalize across diverse datasets and objects in a zero-shot paradigm. But ZIM goes beyond, delivering highly accurate, fine-grained masks that capture intricate details.

🔍 Get Started with ZIM 🔍

Ready to elevate your AI projects with unmatched matting quality? Access ZIM on our [project page](https://naver-ai.github.io/ZIM/), [Arxiv](https://huggingface.co/papers/2411.00626), and [Github](https://github.com/naver-ai/ZIM).

## Installation

```bash
pip install zim_anything
```

or

```bash
git clone https://github.com/naver-ai/ZIM.git
cd ZIM; pip install -e .
```


## Usage

1. Make the directory `zim_vit_l_2092`.
2. Download the [encoder](https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx?download=true) weight and [decoder](https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx?download=true) weight.
3. Put them under the `zim_vit_b_2092` directory.

```python
from zim_anything import zim_model_registry, ZimPredictor

backbone = "vit_l"
ckpt_p = "zim_vit_l_2092"

model = zim_model_registry[backbone](checkpoint=ckpt_p)
if torch.cuda.is_available():
    model.cuda()

predictor = ZimPredictor(model)
predictor.set_image(<image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{kim2024zim,
  title={ZIM: Zero-Shot Image Matting for Anything},
  author={Kim, Beomyoung and Shin, Chanyong and Jeong, Joonhyun and Jung, Hyungsik and Lee, Se-Yun and Chun, Sewhan and Hwang, Dong-Hyun and Yu, Joonsang},
  journal={arXiv preprint arXiv:2411.00626},
  year={2024}
}