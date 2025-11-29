SolarTwin AI: Universal Spatio-Temporal Transformer (USTT)

Official Research Artifact for "Context-Aware Universal Transformers: A Scale-Invariant Framework for Multi-Site Photovoltaic Forecasting".
📌 Project Overview

This repository hosts the Digital Twin Dashboard and inference engine for the Universal Spatio-Temporal Transformer (USTT). This framework addresses the critical challenge of capacity imbalance in multi-site solar forecasting, allowing a single Deep Learning model to predict power generation across heterogeneous sites (e.g., 250 MW Utility Scale vs. 22 MW Residential) without re-training.

Key Innovations

Universal Architecture: Uses a Transformer Encoder with learnable Site Embeddings to switch physical contexts dynamically.

Scale-Invariant Loss: Implements a weighted Huber Loss that inversely weights gradients by plant capacity, achieving $R^2=0.901$ (Plant 1) and $R^2=0.786$ (Plant 2).

Real-Time Efficiency: Optimized for CPU inference with ~1.00 ms latency per sample, meeting microgrid control loop requirements (<50ms).

Physics-Aware Inputs: Utilizes cyclical temporal encoding ($sin(t), cos(t)$) to preserve diurnal solar continuity.

🚀 Live Demo

(https://solartwinai.streamlit.app/) Click Here to Launch the Digital Twin Dashboard 

🛠️ Installation & Local Execution

To reproduce the results or run the digital twin locally:

1. Clone the Repository

git clone [https://github.com/smri29/SolarTwinUstt.git](https://github.com/smri29/SolarTwinUstt.git)
cd SolarTwinUstt


2. Install Dependencies

It is recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


3. Run the Dashboard

streamlit run app.py


Access the app at http://localhost:8501.

📂 Repository Structure

├── app.py                  # Main Streamlit Digital Twin application
├── ustt_model.keras        # Pre-trained Universal Transformer (TF 2.x)
├── scaler.pkl              # Scikit-learn MinMax Scaler (Asset)
├── test_data_sample.csv    # Stratified test set for live inference
├── requirements.txt        # Python dependencies
└── README.md               # Documentation


📊 Model Architecture

The model accepts a time-series tensor $(B, L, D)$ and a site index $(B, 1)$.

Input: 96-step sequence (24 hours) of [Power, Irradiance, Temp, Time_Sin, Time_Cos].

Embedding: Site ID is projected to $d_{model}=64$ and injected into the sequence.

Transformer: 4x Multi-Head Attention Blocks with GELU activation and LayerNorm.

Output: Scalar regression (Next-Step Power).

Note: The model file included (ustt_model.keras) is optimized for CPU inference via float32 casting layers.

🔗 Citation

If you use this code or architecture in your research, please cite the following paper:



📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
