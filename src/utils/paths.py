import os

current_dir = os.path.dirname(os.path.abspath(__file__))
DATABASE_UR = "your_postgresql_database_uri_here"

BASE_DIR = os.path.dirname(current_dir)
EXATA_DIR = os.path.join(BASE_DIR, "_simulation_files")
EXATA_OUTPUT_DIR = os.path.join(EXATA_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "__logs")

MODELS_DIR = os.path.join(BASE_DIR, "saved_modell")

MANAGER_FILE = os.path.join(EXATA_DIR, "manager.conf")
VIEWER_FILE = os.path.join(EXATA_DIR, "viewer.json")

APP_FILE = os.path.join(EXATA_OUTPUT_DIR, "MultiRAT.app")
NODES_FILE = os.path.join(EXATA_OUTPUT_DIR, "MultiRAT.nodes")
CONFIGS_FILE = os.path.join(EXATA_OUTPUT_DIR, "MultiRAT.config")
PATHLOSS_FILE = os.path.join(EXATA_OUTPUT_DIR, "MultiRAT.pathloss")
EXATA_DB_FILE = os.path.join(EXATA_OUTPUT_DIR, "MultiRAT.sqlite")
CONNECTIONS_FILE = os.path.join(EXATA_OUTPUT_DIR, "connection.csv")
CONNECTIOID_FILE = os.path.join(EXATA_OUTPUT_DIR, "connection_id.csv")
TXPOWER_FILE = os.path.join(EXATA_OUTPUT_DIR, "tx_power.csv")


MULTIRAT_PYTHON_PATH = "/home/devruby/dev/ATR-MultiRAT-Manager_v6/.venv/bin/python"
MULTIRAT_MANGER_PATH = "/home/devruby/dev/ATR-MultiRAT-Manager_v6/.venv/bin/manager"
MULTIRAT_QOS_EVAL_PATH = "/home/devruby/dev/ATR-MultiRAT-Manager_v6/.venv/bin/qos_eval_result"
