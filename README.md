# Multi-RAT GNN Modeling

This project includes training scripts and configurations for performance modeling in multi-RAT wireless networks using both graph-based and non-graph machine learning methods.

## 📁 Structure

- `src/train_graph.py`  
  Train various GNN models including:
  - GAT
  - GCN (ATARI)
  - HeGATConv (heterogeneous GAT with attention)
  - MPNN

- `src/train_non_graph.py`  
  Train non-graph models:
  - XGBoost
  - FNN

## 📦 Installation

This project uses [`poetry`](https://python-poetry.org/) for dependency management.  
To install all dependencies:

```bash
poetry install
```

To activate the virtual environment created by Poetry:

```bash
source .venv/bin/activate
```

All required packages are listed in `pyproject.toml`.

## 📊 Dataset

The dataset is generated from the **EXata simulator** for multi-RAT scenarios.

Download the dataset from the following link:

🔗 [Download from Dropbox](https://www.dropbox.com/scl/fo/oy36m6lkv9sz1i991o6dp/ALhvMBmAu7-9kXbUgR7ON4c?rlkey=89joo5ea4abwvly54dioqx4o7&st=o7jwapvu&dl=0)

### ⬇️ Import into PostgreSQL

1. Import the downloaded SQL or CSV data into your PostgreSQL database.
2. Update the following line in `src/utils/paths.py`:

```python
DATABASE_URI = "your_postgresql_database_uri_here"
```

For example:

```python
DATABASE_URI = "postgresql://user:password@localhost:5432/your_database"
```
