from datasets import load_dataset
import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Example script that takes arguments.")

    # Define arguments
    parser.add_argument("--dataset", type=str, required=True, help="name of the dataset")
    parser.add_argument("--save_location", type=str, help="location to save the dataset")
    parser.add_argument("--file_name",type=str, help="file name")

    # Parse the arguments
    args = parser.parse_args()
    print({args.dataset})
    print(args.save_location)
    print(args.file_name)

    # Use the arguments

    # Load the dataset)

    dataset = load_dataset(args.dataset)

    dataset["train"].to_csv(f"{args.save_location}/{args.file_name}_train.csv", index=False)
    dataset["test"].to_csv(f"{args.save_location}/{args.file_name}_test.csv", index=False)

if __name__ == "__main__":
    main()

# Example: neural-bridge/rag-dataset-1200


