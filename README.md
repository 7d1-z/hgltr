# Code for "HGLTR:Hierarchical knowledge injection for calibrating pre-trained models in long-tail recognition"
In this paper, we propose HGLTR (Hierarchy-Guided Long-Tail Recognition)

## Dataset
- Place365: Small images (256 * 256) with easy directory structure - Train and val images. 21G. [[link]](http://places2.csail.mit.edu/download-private.html) [[hierarchical structure]](https://github.com/CSAILVision/places365)
- iNaturalist2018: All training and validation images [120GB] [[link]](https://github.com/visipedia/inat_comp/tree/master/2018#Data)
- ImageNet: Training images (Task 1 & 2). 138GB. & Validation images (all tasks). 6.3GB. [[link]](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) [[details of dataset construction]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR/issues/70) [[hierarchical structure]](https://observablehq.com/@mbostock/imagenet-hierarchy#tree)

## Reproduce
- Dependencies: [requirements.txt](./requirements.txt)
- Bash script: [run.sh](./run.sh)

## Reference
- [openai/CLIP](https://github.com/openai/CLIP)
- [shijxcs/LIFT](https://github.com/shijxcs/LIFT)