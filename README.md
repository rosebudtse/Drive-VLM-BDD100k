# Drive-VLM: Efficient Autonomous Driving Video Understanding via Hybrid-Granularity Alignment

Drive-VLM is an efficient multimodal framework built upon the **SigLIP2-SO400M** vision encoder and **Qwen3-4B** language model. It addresses the "granularity mismatch" in autonomous driving by enabling adaptive switching between detailed perception (**Microscope Mode**) and concise decision reasoning (**Copilot Mode**).

## Project Documentation
- [Full Project Report](./docs/Project_Report.pdf)
- [Presentation Slides](./docs/Presentation_Slides.pdf)

![Drive-VLM Architecture](./docs/assets/architecture.png)

## Key Contributions

- **Hybrid-Granularity Training**: Decouples visual reporting from tactical reasoning through a novel split-entry strategy.

- **Three-Stage Evolution**: Progresses from feature alignment (Stage 1) to instruction tuning (Stage 2) and physics-aware policy optimization (Stage 3).

- **GRPO Policy Optimization**: Implements Group Relative Policy Optimization (GRPO) with a physics-aware reward function to mitigate visual hallucinations.

- **Low-Resource Efficiency**: Achieves effective domain adaptation on a single **NVIDIA RTX 4090 (24GB)** using only 446 curated BDD100k video clips.



## Three-Stage Training Strategy

| Stage | Name | Description | Trainable Modules |
| --- | --- | --- | --- |
| **Stage 1** | Projector Pre-training | Aligns SigLIP2 visual features with Qwen3 token space. | MLP Projector, Temporal Embeddings  |
| **Stage 2** | Split-Entry SFT | Teaches the model to switch between "Microscope" and "Copilot" granularities. | LoRA Adapters, MLP Projector |
| **Stage 3** | Physics-Aware GRPO | Uses YOLO-based verification to enforce physical consistency and suppress hallucinations. | LoRA Adapters |


## Physics-Aware Reward Function (Stage 3)

To ensure safety and reliability, our GRPO phase utilizes a multi-metric reward :
- **$R_{obj}$ (Object Grounding)** *(weight: 0.40)*: F1-score comparison against YOLOv8x detections.

- **$R_{temp}$ (Temporal Consistency)** *(weight: 0.25)*: Assesses causal progression and dynamic verb density.

- **$R_{detail}$ (Detail Richness)** *(weight: 0.20)*: Encourages comprehensive descriptions

- **$R_{hall}$ (Hallucination Penalty)** *(weight: 0.15)*: Strictly penalizes mentions of safety-critical objects absent in visual evidence.




## Repository Structure

```text
├── assets/
├── docs/
│   ├── IERG5050_Project_Report.pdf      # Full project report
│   └── IERG5050_Presentation.pptx       # Presentation slides
├── model_new.py                         # Architecture: SigLIP2-SO400M + Qwen3-4B
├── dataset.py                           # BDD100k VideoInstructDataset with Split-Entry logic
├── train_stage1_ddp.py                  # Visual features and token spacd alignment
├── train_stage2_new.py                  # Hybrid-Granularity SFT
├── train_stage3.py                      # Physics-Aware Policy Optimization (GRPO)
├── check_dataset.py                     # Diagnostic tool for masking (-100) and EOS tokens
├── stage3/                              # Reward functions and YOLO-based object verification
│   ├── grpo_reward_func.py
│   └── vocabulary_config.py
├── output/                              # Training logs and visualized curves (images, runs)
└── README.md                              

```


## Experimental Results

### Qualitative Performance

The transition from Stage 2 to Stage 3 shows a significant reduction in subjective "hallucinations" and an increase in actionable driving insights.

| Mode | Stage 2 (SFT) | Stage 3 (GRPO) |
| --- | --- | --- |
| **Microscope** | Descriptive, includes subjective fillers. | Object-centric, verified by physical detection. |
| **Copilot** | Verbose sentence structure. | Telegraphic, actionable insights. |


## Installation & Usage

### Prerequisites

* Python 3.10+
* PyTorch 2.x
* NVIDIA GPU with 24GB+ VRAM (RTX 4090 recommended) 



### Run Validation

Before training, verify the data masking and label integrity:

```bash
python check_dataset.py

```


## 📝 Author

XIE Zifan, MSc in Artificial Intelligence, The Chinese University of Hong Kong (CUHK).

