# Textual Enhanced Contrastive Learning for Solving Math Word Problems.
Shen Y.\*, Liu Q.\*, Mao Z., Cheng F., Sadao K. (2022)

This paper has been accepted for publication in *Findings of EMNLP 2022*.

The arxiv preprint could be found [here](https://arxiv.org/abs/2211.16022).

Solving math word problems is the task that analyses the relation of quantities and requires an accurate understanding of contextual natural language information. Recent studies show that current models rely on shallow heuristics to predict solutions and could be easily misled by small textual perturbations. To address this problem, we propose a Textual Enhanced Contrastive Learning framework, which enforces the models to distinguish semantically similar examples while holding different mathematical logic. We adopt a self-supervised manner strategy to enrich examples with subtle textual variance by textual reordering or problem re-construction. We then retrieve the hardest to differentiate samples from both equation and textual perspectives and guide the model to learn their representations. Experimental results show that our method achieves state-of-the-art on both widely used benchmark datasets and also exquisitely designed challenge datasets in English and Chinese. 


## Data

Math23K data used for this paper could be found in /data folder. The HE-MWP data is in preparation and is coming soon.


## Code

### Math23K Reproduction

For reproducing Math23K results, please run:

```
python main.py
```


## Citation



If you find this repo useful, please cite the following paper:

```
@misc{https://doi.org/10.48550/arxiv.2211.16022,
  doi = {10.48550/ARXIV.2211.16022},
  url = {https://arxiv.org/abs/2211.16022},
  author = {Shen, Yibin and Liu, Qianying and Mao, Zhuoyuan and Cheng, Fei and Kurohashi, Sadao},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Textual Enhanced Contrastive Learning for Solving Math Word Problems},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
