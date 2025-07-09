# Violent-object-and-action-recognition-using-YOLO-and-Hyperformer
This project was developed as part of a university research initiative focused on human action recognition, with a particular emphasis on detecting violent actions. It explores the integration of various AI models to preprocess and enrich data for input into Hyperformer, a novel hypergraph-based Transformer architecture designed for skeleton-based action classification.

The pipeline combines multiple components to process video data and extract meaningful information:
  - YOLO – Used in two instances:
    - One for person detection and tracking
    - Another for weapon detection and tracking, providing contextual cues that are later fused with pose data
  - DepthAnything – Employed to cluster nearby individuals using depth estimation, which helps in analyzing interactions in crowded scenes
  - MediaPipe – Utilized to extract human skeletons from video frames, which are then transformed into sequences for Hyperformer

The extracted skeletal data is used as the primary input to the Hyperformer model. Additional features—such as weapon presence and proximity between individuals—are incorporated into the model to assess whether such multimodal information improves classification accuracy.


The core of the work involves building a custom dataset, preprocessing it through this multi-stage AI pipeline, and evaluating the effect of integrating visual context (e.g., weapons, spatial clustering) on the model’s performance in classifying violent actions.

# Hyperfomer Evaluation with YOLO-Weapon Integration
## 📁 Dataset Preparation

To process the raw data and build the dataset:

1. Navigate to the folder: `Data/Processed_data`
2. Execute the following scripts in order:
   - `get_raw_skes_data.py`
   - `get_raw_denoised_data.py`
   - `seq_transformation.py`

The final output will be a dataset named: `FTVD.npz`

---

## 🧪 Models Training & Evaluation

### 1. Move to the Model Folder

```bash
cd Hyperformer
```

### 2. Fine-Tuning Models

- **With YOLO-Weapon integration**:  
  Run the appropriate script:

  ```bash
  ./finetune_Hyperf*.sh
  ```

- **Without YOLO-Weapon integration**:  
  Run the script:

  ```bash
  ./finetune_Hyperf*_noYolo.sh
  ```

> Replace `*` with the corresponding model number (e.g., `1`, `2`, `3`).

---

### 3. Evaluation Instructions

Before running the evaluation:

#### 🔧 Modify `main.py` Files

- **For `Hyperf_1` and `Hyperf_1_noYolo`**:
  - Comment out **lines 358–360**

- **For `Hyperf_2`, `Hyperf_2_noYolo`, `Hyperf_3`, and `Hyperf_3_noYolo`**:
  - Comment out **lines 365–369**

#### 📝 Edit `evaluate.sh` Script

- **Line 12** – Set the desired weight file path:
  ```bash
  weights=work_dir/Hyperf_*/pretrained/your_weight.pth
  ```

- **Line 15** – Specify the correct config file:
  ```bash
  config=config/your_model_joint.yaml
  ```

- **Line 16** – Set the model name:
  ```bash
  model=YourModel.Model
  ```

- **Line 17** – Define the output directory:
  ```bash
  work_dir=work_dir/Hyperf_*/your_custom_dir
  ```
