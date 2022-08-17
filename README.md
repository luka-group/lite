# LITE:beers:
This is the repository for the resources in TACL 2022 Paper "Ultra-fine Entity Typing with Indirect Supervision from Natural Language Inference". This repository contains the source code and links to some pre-trained model checkpoint.

## Abstract
The task of ultra-fine entity typing (UFET) seeks to predict diverse and free-form words or phrases that describe the appropriate types of entities mentioned in sentences. A key challenge for this task lies in the large amount of types and the scarcity of annotated data per type. Existing systems formulate the task as a multi-way classification problem and train directly or distantly supervised classifiers. This causes two issues: 
  (i) the classifiers do not capture the type semantics since types are often converted into indices; (ii) systems developed in this way are limited to predicting within a pre-defined type set, and often fall short of generalizing to types that are rarely seen or unseen in training. 

This work presents LITE:beers:, a new approach that formulates entity typing as a natural language inference (NLI) problem, making use of (i) the indirect supervision from NLI to infer type information meaningfully represented as textual hypotheses and alleviate the data scarcity issue, as well as (ii) a learning-to-rank objective to avoid the pre-defining of a type set. Experiments show that, with limited training data, LITE obtains state-of-the-art performance on the UFET task. In addition, LITE demonstrates its strong generalizability, by not only yielding best results on other fine-grained entity typing benchmarks, more importantly, a  pre-trained LITE system works well on new data containing unseen types.

![Fig1 in paper](https://github.com/luka-group/lite/blob/main/readme/lite.png)

## Environment

    python 3.7
    Transformers (Huggingface) version 4.6.1 (Important)
    PyTorch with CUDA support
    cudatoolkit 10.0
    CUDA Version 11.1
  
## Dataset  
The Ultra-fine Entity Typing (UFET) dataset is available at https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html. 


## Run the experiment  
1. Download and unzip UFET data. Move its /release directory to /data
2. Run data/process_ultrafine.py to pre-process UFET data
3. Edit and run lite.sh to run the experiment for training.
4. Edit and run eval.sh to generate type vocab rankings given their confidence score
5. Edit and run result.sh to calculate the best threshold on dev set and generate the typing result on test set given such threshold.


## Link to the pre-trained full models and results
**Pre-trained LITE checkpoint** is available at https://drive.google.com/file/d/1gICYx_UzHGcRNg3k-DPNx9w0JJKHg4AR/view?usp=sharing for users to do inference on their own data.
**Result** of our pre-trained LITE is available at https://drive.google.com/file/d/1c02mOh_dozJq7afeUwGfXXn-cXcDV9vq/view?usp=sharing.

**Out-of-the-Box version on CoLab**
In progress...

## Reference
Bibtex:
  
    @article{li-etal-2022-lite,
      title={Ultra-fine Entity Typing with Indirect Supervision from Natural Language Inference},
      author={Li, Bangzheng and Yin, Wenpeng and Chen, Muhao},
      journal={Transactions of the Association for Computational Linguistics},
      volume={10},
      year={2022},
      publisher={MIT Press}
    }


This project is supported by by the National Science Foundation of United States Grant IIS 2105329.
