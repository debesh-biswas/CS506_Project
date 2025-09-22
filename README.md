# CS506 Project Proposal  
**Project Title:** Asteroid Tracking & Impact Risk Prediction  

---

## 1. Project Description  
We’re building a data science pipeline to study **near-Earth objects (NEOs)** and see which ones might be risky for Earth.  
NASA and JPL publish open datasets with orbital parameters, close-approach info, and physical stats of asteroids.  
Our idea is to collect this data, clean it up, make useful features, and then train models to predict whether an asteroid is hazardous.  

The final deliverable won’t just be numbers we’ll also build an **interactive dashboard** so people can explore orbits, risks, and asteroid stats visually.  

---

## 2. Project Goals  
- Collect and clean asteroid orbital + physical data.  
- Create features related to risk (MOID, velocity, estimated size, etc.).  
- Train and test models for hazardous vs. non-hazardous classification.  
- Visualize distributions and orbital paths.  
- Build an interactive dashboard with risk scores.  
- Keep everything version-controlled and reproducible in GitHub.  

---

## 3. Data Collection Plan  
We’ll mainly use:  
- **NASA NeoWs API** → orbital parameters, size estimates, hazard flags.  
- **JPL Small-Body Database (SBDB)** → detailed orbital elements + close approaches.  

**Method:**  
- Write Python scripts to query the APIs.  
- Save results in CSV for reproducibility.  
- Aim for ~5k–10k asteroids from the last decade.  

---

## 4. Data Cleaning Plan  
- Deal with missing values + inconsistent formats.  
- Standardize units (distances → km, velocities → km/s).  
- Encode hazard flag as binary.  
- Remove duplicates and incomplete records.  

---

## 5. Feature Extraction  
Some features we’ll use:  
- **MOID (Minimum Orbit Intersection Distance)** → how close the orbit comes to Earth.  
- **Relative velocity** at closest approach.  
- **Estimated diameter** (from absolute magnitude + albedo).  
- **Observation arc length** → proxy for uncertainty.  
- **Close-approach frequency** → how often it gets near Earth.  

---

## 6. Data Visualization  
Planned visuals:  
- Scatter plots: asteroid size vs. velocity, colored by hazard flag.  
- Histograms: MOID distributions.  
- Interactive 3D orbit plots (Plotly/Dash).  
- Dashboard view: top 10 “riskiest” asteroids with key stats.  

---

## 7. Modeling Plan  
We’ll start simple and then try more advanced models:  
- **Baseline:** Logistic Regression.  
- **Tree-based:** Decision Tree, Random Forest, XGBoost.  
- **Optional:** Neural nets if time allows.  

**Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC.  

---

## 8. Test Plan  
- Split data: 80% train / 20% test.  
- Use stratified sampling for class balance.  
- Do 5-fold cross-validation.  
- Final evaluation on a held-out test set.  

---

## 9. Expected Outcomes  
By the end, we expect:  
- A cleaned dataset of asteroid features.  
- Trained classification models with evaluation metrics.  
- An interactive dashboard to explore asteroid risks.  
- A reproducible GitHub repo with code + documentation.  

