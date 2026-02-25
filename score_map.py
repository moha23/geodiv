import pandas as pd
import os
import argparse

axes = ["affluence", "maintenance", "bgr","obj"]
datasets = ["sd21", "sd3m", "sd35", "flux1"]
entities = ["bag", "backyard", "car", "chair", "dog", "house", 
            "storefront", "stove", "cooking pot", "plate of food"]

# Define country list in correct order
countries = [
    "United Kingdom", "Nigeria", "Turkey", "Indonesia", "Colombia",
    "Japan", "India", "China", "United States of America", "Mexico",
    "United Arab Emirates", "South Korea", "Spain", "Italy",
    "Egypt", "Philippines"
]

def main(args):
    
    for axis in axes:

        axis_df_all = []

        for dataset in datasets:
            for entity in entities:

                path = f"{args.output_path}/{dataset}/{entity}/hillno_scores_{axis}.csv"

                if not os.path.exists(path):
                    continue
                
                print(f"Processing {path}")
                try:
                    df = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    print("Empty file:", path)
                    continue
    
                # Extract country from prompt
                # Prompt format: "A photo of a <entity> in <country>"
                df["country"] = df["prompt"].str.replace(
                    f"A photo of a {entity} in ", "", regex=False
                )

                # Keep relevant columns
                df_small = df[["dataset", "entity", "country", "Normalised D_q"]]

                # Pivot: countries become columns
                df_pivot = df_small.pivot_table(
                    index=["dataset", "entity"],
                    columns="country",
                    values="Normalised D_q"
                ).reset_index()

                axis_df_all.append(df_pivot)

        # Concatenate all dataset/entity combos
        final_axis_df = pd.concat(axis_df_all, ignore_index=True)

        # Ensure country column order
        final_axis_df = final_axis_df[["dataset", "entity"] + countries]

        # Save
        # Fill missing values with NaN for empty files
        final_axis_df = final_axis_df.fillna(0)  # or use other fill strategy
        final_axis_df.to_csv(f"{args.output_path}/hillno_map_{axis}.csv", index=False)

        print(f"Saved hillno_map_{axis}.csv")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run VQA.")
    parser.add_argument("--vqa_model",
                        choices=['qwen', 'flash', 'gpt'],
                        default='flash',
                        help="VLM to use")    
    parser.add_argument("--output_path", type=str, required=False, 
                        default="./results/", 
                        help="Path to the output directory to save the scores.")

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path, args.vqa_model)

    main(args)