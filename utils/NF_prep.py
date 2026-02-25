import pandas as pd
import argparse
import os

def main(args):

    print("Cleaning and preparing the NF flagged questions for VDI VQA \n Entity:", args.entity_name, "\nDataset:", args.dataset_name)
    res_path = args.output_path
    interm_dir = os.path.join('data', args.vqa_model, args.entity_name) # to save intermediate csvs

    df = pd.read_csv(res_path + '/visibility_predictions.csv')
    print('Total number of rows in visibility test:', len(df))
    print('Unique answers in visibility test:')
    print(df['answer'].value_counts())
    
    # standardizing the answer format for yes
    df['answer'] = df['answer'].replace({'["yes"]': "['yes']"})
    df_yes = df[df['answer']=="['yes']"].copy()
    
    df_whole = pd.read_csv('data/all.csv')
    df_whole=df_whole[['entity_id','entity','prompt_id','prompt','question_id','question','answer_list','flag']]
    
    df_F = df_whole[df_whole['flag']=='F']
    df_F.to_csv('data/vqa_F.csv', index=False)
    
    df_whole['question_id'] = df_whole['question_id'].astype(str)
    df_yes['question_id'] = df_yes['question_id'].astype(str)

    # Keep df_yes, replace questions and attributes in it with values from df_whole as per correspondence with question_id and prompt_id
    # Merge with df_whole, which has the updated 'question' and 'answer_list'
    df_updated = df_yes.merge(
        df_whole[['question_id', 'prompt_id', 'question', 'answer_list']],
        on=['question_id', 'prompt_id'],
        how='left',
        suffixes=('', '_from_whole')  # This avoids overwriting during merge
    )

    # Overwrite df_yes's values with the ones from df_whole
    df_updated['question'] = df_updated['question_from_whole']
    df_updated['answer_list'] = df_updated['answer_list_from_whole']

    # Drop the temporary columns
    df_updated.drop(columns=['question_from_whole', 'answer_list_from_whole'], inplace=True)


    df_updated['image_name'] = df_updated['image_path'].apply(lambda x: x.split('/')[-1].replace(".png", ""))
    df_updated['custom_id'] = df_updated.apply(lambda x: f"{x['entity_id']}_{x['question_id']}_{x['prompt_id']}_{x['image_name']}", axis=1)
    df_updated = df_updated[['entity_id', 'entity', 'prompt_id', 'prompt', 'question_id', 'question', 'answer_list', 'image_path', 'custom_id']]

    print('Total number of rows for NF flagged questions', len(df_updated))
    print('Saved updated dataframe with NF flagged questions and image paths to csv at:', interm_dir+'/vqa_NF_'+args.axis+'_'+args.entity_name+'_'+args.dataset_name+'_with_images.csv')
    df_updated.to_csv(interm_dir+'/vqa_NF_'+args.axis+'_'+args.entity_name+'_'+args.dataset_name+'_with_images.csv', index=False)



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
                        default='entity',
                        help="Name of the VDI axis you want to assess.")

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path, args.vqa_model, args.dataset_name, args.entity_name)

    main(args)