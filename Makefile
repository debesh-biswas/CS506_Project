# Cross-platform Makefile focused on Windows cmd.exe
# Orchestrates reproducibility steps described in README.

.PHONY: all setup clean_data impute engineer model figures clean_env nbexec

# Default pipeline: setup env, clean raw, impute missing, engineer features, then model
all: setup clean_data impute engineer model

# Create virtual environment and install dependencies without relying on activation
setup:
	python -m venv .venv
	".venv\\Scripts\\python" -m pip install --upgrade pip
	".venv\\Scripts\\python" -m pip install -r requirements.txt
	".venv\\Scripts\\python" -m pip install jupyter

# Step 1 – Cleaning (produces Dataset/neo_clean.csv)
clean_data:
	".venv\\Scripts\\python" "Dataset Processing\\clean.py"

# Step 2 – Handling Missing Data (produces Dataset/neo_model.csv)
# Execute the notebook headlessly using nbconvert
impute: nbexec
	".venv\\Scripts\\python" -m jupyter nbconvert --to notebook --execute "Dataset Processing\\missing_data_handle.ipynb" --output "missing_data_handle.executed.ipynb"

# Step 3 – Feature Engineering (produces Dataset/neo_processed.csv)
engineer:
	".venv\\Scripts\\python" "Dataset Processing\\feature_engineer.py"

# Step 4/5 – Visualization & Modeling (final notebook)
# Execute the final modeling notebook headlessly as well
model: nbexec
	".venv\\Scripts\\python" -m jupyter nbconvert --to notebook --execute "Notebooks\\Data_Modelling Final.ipynb" --output "Data_Modelling Final.executed.ipynb"

# Optional: place to collect figures (if scripts are added later)
figures:
	@echo [figures] Figures are saved during notebook runs into the \"Figures\" folder

# Remove virtual environment (use with care)
clean_env:
	@echo [clean_env] Removing .venv
	@if exist ".venv" rmdir /S /Q .venv

# Internal: ensure headless notebook execution prerequisites exist
nbexec:
	@echo [nbexec] Using jupyter nbconvert for headless notebook execution
