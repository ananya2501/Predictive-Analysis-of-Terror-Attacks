This project applies machine learning to historical terrorism data to understand spatio-temporal patterns, identify important factors, and generate early-warning risk indicators at an aggregated regional level. The aim is purely academic and research-focused, emphasizing transparency, ethical handling of sensitive data, and explainable AI.The repository includes complete workflows: data preprocessing, feature engineering, EDA, model training, explainability, evaluation, and interactive tools built with Streamlit.

🚀 Key Features 
📊 Exploratory Data Analysis (EDA) with visualizations 
🧹 Data preprocessing and feature engineering 
🤖 Machine Learning models (Logistic Regression, Random Forest, XGBoost, SARIMA/Prophet) 
📈 Forecasting of regional monthly attack counts 
🔍 Explainability using SHAP, feature importance & PDPs 
🧭 Streamlit Dashboard for interactive exploration 
🎛️ Gradio Demo for quick model predictions 
🧪 Temporal validation & performance evaluation 
🛡️ Ethics, limitations & bias assessment


📂 Dataset 
This project uses publicly available research datasets, such as: Global Terrorism Database (GTD) Data is historical, aggregated, and used strictly for academic analysis. Coordinates are generalized or region-level only.



🧠 Methods & Approach
Data Processing Cleaning & handling missing values Converting incident-level data into spatio-temporal aggregates Generating lag features, seasonal indicators & contextual variables
Modeling Techniques Baselines: Naïve persistence, Logistic/Poisson regression ML models: Random Forest, XGBoost Time-series: Prophet / SARIMA
Model Evaluation Temporal train-test split Metrics: Precision, Recall, F1, AUPRC, RMSE Calibration curves Error analysis & uncertainty estimation
Explainability SHAP value plots Feature importance Partial dependence plots Region & time-specific explanations


🖥️ Interactive Applications 
🔵 Streamlit Dashboard 
Visualizes: 
Historical trends Regional risk indicators Time-series forecasts SHAP explanations Maps (generalized regions only)


Launch using: streamlit run dashboards/streamlit_app.py
Run using: python dashboards/gradio_demo.py


⚙️ Installation git clone https://github.com/your-username/predictive-terror-analysis.git cd predictive-terror-analysis pip install -r requirements.txt


▶️ Usage
Preprocess the dataset python src/preprocess.py
Train ML models python src/train.py
Run evaluation & generate explainability plots python src/evaluate.py
Launch dashboards


🛡️ Ethical Considerations This project is strictly for academic research and educational purposes. It does not aim to predict specific events, identify individuals, or support operational decision-making.


Key considerations: Use only public, licensed datasets Aggregate data to region-level; never use sensitive coordinates Acknowledge data bias (media, underreporting, geography) Include uncertainty in predictions Avoid interpretations that can harm groups or regions


🤝 Contributions

Contributions, issues, and feature suggestions are welcome! Feel free to submit pull requests or open discussions.
