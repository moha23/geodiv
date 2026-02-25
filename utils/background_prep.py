import pandas as pd
import argparse
import os

answer_map= {
    # Indoor Questions
    "What best describes the visual order in this image?":
        "['Organized (several elements present, but neat, intentional arrangement)', 'Cluttered (many elements, visually noisy, no clear order)', 'Minimalist (very few or no elements at all, mostly empty or plain)']",
    
    "Which main elements are visible in the background?":
        "['Walls', 'Windows', 'Furniture', 'Appliances (e.g. fridge, microwave, washing machine)', 'Electronic equipment (e.g. TVs, computers, speakers)', 'Plain / Solid color background']",
     
    "What type of environment is visible?":
        "['Residential', 'Commercial / Public', 'Plain / Solid color background']",
    
    "What type of floor or ground is visible?":
        "['Tiled Floor', 'Wooden Floor', 'Carpeted Floor', 'Concrete Floor']",

    # Outdoor Questions
    "What type of background elements are most visible?":
        "['Natural (trees, sky, soil, water, mountains)','Built structures (walls, windows, houses, buildings, fences)', 'Mixed (both natural and built elements visible)']",
        
    "How dense is the built environment in the background?":
        "['Sparse / Open (fields, wide spaces, few or no buildings)', 'Moderate (some houses/buildings, not crowded)', 'Dense / Crowded (clustered buildings, narrow streets, crowded interiors)']", 
    
    "What type of modern infrastructure is visible in the background?":
        "['Transport-related (paved roads, vehicles, bridges, rail tracks)', 'Utility-related (electric poles, wires, water tanks, pipelines)', 'High-rise / Industrial (skyscrapers, factories, construction sites, large machinery)']",
        
    "How busy does the background appear, crowded (many people, vehicles, signs of activity), moderately busy (some human activity), or quiet / empty (few or no people or vehicles)?":
        "['Crowded', 'Moderately busy', 'Quiet / Empty']",

    "What natural features, if any, are visible in the background of the image?":
        "['trees / forest / plants', 'mountains / hills', 'waterbody', 'open ground / fields']",
    
    "What type of road or terrain is visible?":
        "['Paved road', 'Dirt / gravel road (man-made)', 'Natural ground / grass (wild, non-constructed)', 'Tiled / courtyard-style surface']"
}

question_map= {
    # Indoor Questions
    "0i":"What best describes the visual order in this image?",
    
    "1i":"Which main elements are visible in the background?",
    
    "2i":"What type of environment is visible?",
    
    "3i":"What type of floor or ground is visible?",

    # Outdoor Questions
    "0o":"What type of background elements are most visible?",
    
    "1o":"How dense is the built environment in the background?",
    
    "2o":"What type of modern infrastructure is visible in the background?",
    
    "3o":"How busy does the background appear, crowded (many people, vehicles, signs of activity), moderately busy (some human activity), or quiet / empty (few or no people or vehicles)?",

    "4o":"What natural features, if any, are visible in the background of the image?",
    
    "5o":"What type of road or terrain is visible?"
}

visibility_check_map_indoor = {
    # Indoor Questions
    "0i":
        "Is the layout of the indoor space (apart from the entity) visible enough to judge how visually organized it appears?",    
    "1i":
        "Is the background beyond the main entity visible enough to judge which elements (e.g., walls, furniture, appliances) are present or absent?",    
    "2i":
        "Is the broader indoor setting beyond the entity visible enough to infer if it's residential, commercial, or public?",    
    "3i":
        "Is the floor or ground clearly visible in the image, excluding any surface that is part of the entity itself?",
}

visibility_check_map_outdoor = {
    # Outdoor Questions
    "0o":
        "Is the background beyond the main entity visible enough to distinguish whether it contains natural scenery, built structures, or both?",    
    "1o":
        "Is the background clear enough beyond the main entity to judge the level of building density (sparse, moderate, or dense)?",    
    "2o":
        "Is the background visible enough beyond the main entity to judge whether modern infrastructure (e.g., roads, vehicles, electric poles, large buildings) is present or absent?",    
    "3o":
        "Is the background visible enough beyond the main entity to judge how busy the scene appears (crowded, moderately busy, or quiet/empty)?",    
    "4o":
        "Is the background visible enough beyond the main entity to judge whether natural features such as trees, waterbodies, or hills are present or absent?",    
    "5o":
        "Is the ground or road surface in the background visible enough (excluding any surface that belongs to the main entity) to judge its type?"} 

def clean_json(ans): 
    # ans = row['answer']
    if isinstance(ans, str) and 'json' in ans.lower():
        print(f"Current value: {ans}")
        new_val = ans.replace("```json","").replace("```","").replace("\n","").strip()
        print(f"Saved {new_val}")
        print()
        return new_val
    return ans

def main(args):
    print()
    print("==========Background VQA Clean Up==============")
    if args.step_name=='step2':
        print("Cleaning after step 1...")
        df = pd.read_csv(f'{args.output_path}/step1_predictions.csv')
        print("Before cleaning:")
        print(df['answer'].value_counts())
        df.loc[df['answer'].str.contains('yes'), 'answer'] = '["yes"]'
        df_step2=df[df['answer']=='["yes"]']
        print("After cleaning, number of rows left:", len(df_step2))
        df_step2.loc[:, 'question'] = "Does the setting appear to be indoors or outdoors?"
        df_step2.loc[:, 'answer_list'] = "['indoors', 'outdoors']"
        df_step2 = df_step2.drop(columns=['reasoning_steps', 'answer'])
        df_step2.to_csv(f'data/{args.vqa_model}/{args.entity_name}/step2_background_{args.entity_name}_{args.dataset_name}_with_images.csv', index=False)
        print("Cleaned data for step 2 saved to:")
        print(f'data/{args.vqa_model}/{args.entity_name}/step2_background_{args.entity_name}_{args.dataset_name}_with_images.csv')
    
    elif args.step_name=='step3':
        print("Cleaning after step 2...")
        df = pd.read_csv(f'{args.output_path}/step2_predictions.csv')
        print("Before cleaning:")
        print(df['answer'].value_counts())
        df.loc[df['answer'].str.contains('indoors'), 'answer'] = "['indoors']"
        df.loc[df['answer'].str.contains('outdoors'), 'answer'] = "['outdoors']"
        rows = []
        for _, row in df[df['answer'] == "['indoors']"].iterrows():
            for qid, question in visibility_check_map_indoor.items():
                new_row = row.copy()
                new_row['question_id'] = qid
                new_row['question'] = question
                new_row['answer_list'] = "['yes', 'no']"
                rows.append(new_row)
        for _, row in df[df['answer'] == "['outdoors']"].iterrows():
            for qid, question in visibility_check_map_outdoor.items():
                new_row = row.copy()
                new_row['question_id'] = qid
                new_row['question'] = question
                new_row['answer_list'] = "['yes', 'no']"
                rows.append(new_row)  
        df_background = pd.DataFrame(rows)
        print("After cleaning, number of rows (4*indoor + 6*outdoor):", len(df_background))
        df_background.to_csv(f'data/{args.vqa_model}/{args.entity_name}/step3_background_{args.entity_name}_{args.dataset_name}_with_images.csv', index=False)
        print(f'data/{args.vqa_model}/{args.entity_name}/step3_background_{args.entity_name}_{args.dataset_name}_with_images.csv')
        
    elif args.step_name=='step4':
        print("Cleaning after step 3...")
        df = pd.read_csv(f'{args.output_path}/step3_predictions.csv')
        print("Before cleaning:")
        print(df['answer'].value_counts())
        df.loc[df['answer'].str.contains('yes'), 'answer'] = "['yes']"
        df = df[df['answer']=="['yes']"]
        # Assuming your dataframe has a column with question IDs
        df['question'] = df['question_id'].map(question_map)
        df['answer_list']=df['question'].map(answer_map)
        print("After cleaning:", len(df))
        df = df.drop(columns=['reasoning_steps', 'answer'])
        df.to_csv(f'data/{args.vqa_model}/{args.entity_name}/step4_background_{args.entity_name}_{args.dataset_name}_with_images.csv', index=False)
        print(f"Cleaned and saved to:")
        print(f'data/{args.vqa_model}/{args.entity_name}/step4_background_{args.entity_name}_{args.dataset_name}_with_images.csv')
        
    else:
        print("Cleaning after step 4...")
        df = pd.read_csv(f'{args.output_path}/step4_predictions.csv')
        print("Before cleaning:")
        print(df['answer'].value_counts())
        df['answer']=df['answer'].apply(lambda x: clean_json(x))
        print("\nAfter cleaning:")
        print(df['answer'].value_counts())
        df.to_csv(f'{args.output_path}/step4_predictions.csv', index=False)
        print(f"Cleaned and saved to:")
        print(f'{args.output_path}/step4_predictions.csv')
        
        
    print("==========Done==============")
        
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run VQA.")
    parser.add_argument("--vqa_model",
                        choices=['qwen', 'flash', 'gpt'],
                        default='qwen',
                        help="VLM to use")
    parser.add_argument("--entity_name",
                        choices=['house', 'bag', 'backyard', 'car', 'cooking pot', 'plate of food', 'dog', 'storefront', 'chair', 'stove'],
                        default='house',
                        help="The entity to work on.")    
    parser.add_argument("--output_path", type=str, required=False, 
                        default="./results/", 
                        help="Path to the output directory to save the scores.")
    parser.add_argument("--dataset_name", type=str, required=False, default="sd21",
                        choices=['geode', 'sd21', 'sd3m', 'flux1', 'sd35'],
                        help="Name of the image dataset to be used/stored.")
    parser.add_argument("--axis",
                        choices=['entity', 'background'],
                        default='background',
                        help="Name of the VDI axis you want to assess.")
    parser.add_argument("--step_name",
                        type=str,
                        choices=['step2', 'step3', 'step4', 'final'],
                        default='step2',
                        help="VQA step name.")

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path, args.vqa_model, args.dataset_name, args.entity_name)

    main(args)
    