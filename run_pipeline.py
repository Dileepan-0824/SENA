import subprocess
import sys
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_script(path):
    logging.info(f"Running {path}...")
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = "."
    result = subprocess.run([sys.executable, path], capture_output=True, text=True, env=my_env)
    if result.returncode != 0:
        logging.error(f"Error in {path}:\n{result.stderr}")
        sys.exit(1)
    logging.info(result.stdout.strip())

def main():
    logging.info("Starting  Multiplex Brokerage Drift Pipeline...\n")
    
    scripts = [
        "src/ingest/ingest_pipeline.py",
        "src/alignment/multiplex_alignment.py",
        "src/graphs/graph_builder.py",
        "src/analysis/sena_diagnostics.py",
        "src/modeling/target_builder.py",
        "src/modeling/train.py",
        "src/modeling/replicate_visualizations.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    logging.info("Pipeline Execution Completed.")

if __name__ == "__main__":
    main()
