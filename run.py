import sys
sys.path.insert(0, 'src')

from ehp.pipeline import run_full_analysis

def main():
    try:
        print("Starting pipeline run ...")
        run_full_analysis('data/raw', 'results')
        print("Pipeline run finished successfully.")
    except Exception as e:
        print(f"Pipeline run failed with error: {e}")

if __name__ == '__main__':
    main()
