import argparse
import math
import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from utils.utils import convert_gcs_to_local

def analyze_csv_vs_images(csv_file_path, dataset_name, axis, show_low_coverage=False):
    """
    Analyze CSV file and count rows vs actual images in folders
    """

    df = pd.read_csv(csv_file_path) 
    grouped = df.groupby(['prompt', 'question'])
    
    results = []
    folder_cache = {}  # Cache to avoid counting same folder multiple times
    
    
    for (prompt, question), group in grouped:

        csv_row_count = len(group)
        sample_path = group['image_path'].iloc[0]
        
        local_path = convert_gcs_to_local(sample_path, dataset_name)
        
        folder_path = '/'.join(local_path.split('/')[:-1])
        
        # Count images in folder (with caching)
        if folder_path in folder_cache:
            image_count = folder_cache[folder_path]
        else:
            try:
                if os.path.exists(folder_path):
                    # Count image files
                    image_extensions = {'.png'}
                    files = os.listdir(folder_path)
                    image_count = sum(1 for f in files 
                                    if os.path.splitext(f.lower())[1] in image_extensions)
                    folder_cache[folder_path] = image_count
                else:
                    image_count = 0
                    folder_cache[folder_path] = 0
                    print(f"Warning: Folder not found: {folder_path}")
            except Exception as e:
                image_count = 0
                folder_cache[folder_path] = 0
                print(f"Error accessing folder {folder_path}: {e}")
        
        # Check if CSV rows are less than threshold% of images
        if axis=='entity':
            threshold_coverage = 0.5
        else:
            threshold_coverage = 0.3
        is_low_coverage = csv_row_count < (image_count * threshold_coverage) if image_count > 0 else False
        coverage_percent = (csv_row_count / image_count * 100) if image_count > 0 else 0
        
        results.append({
            'prompt': prompt,
            'question': question, 
            'csv_rows': csv_row_count,
            'image_count': image_count,
            'coverage_percent': coverage_percent,
            'low_coverage': is_low_coverage,
            'folder_path': folder_path,
            'original_gcs_path': sample_path
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    if axis=='entity':
        threshold_coverage = 0.5
    else:
        threshold_coverage = 0.3
        
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total unique prompt+question combinations: {len(results_df)}")
    print(f"Combinations with < {threshold_coverage*100}% coverage: {results_df['low_coverage'].sum()}")
    
    # Show low coverage cases
    if show_low_coverage:
        
        low_coverage_df = results_df[results_df['low_coverage']]
        if len(low_coverage_df) > 0:
            print(f"\n=== LOW COVERAGE CASES (<{threshold_coverage*100}%) ===")
            print(f"Found {len(low_coverage_df)} cases with less than {threshold_coverage*100}% coverage:")
            
            for _, row in low_coverage_df.head(10).iterrows():
                print(f"\nPrompt: {row['prompt']}")
                print(f"Question: {row['question']}")
                print(f"CSV rows: {row['csv_rows']}, Images: {row['image_count']}")
                print(f"Coverage: {row['coverage_percent']:.1f}%")
                print(f"Folder: {row['folder_path']}")
                print("-" * 60)
                
            if len(low_coverage_df) > 10:
                print(f"... and {len(low_coverage_df) - 10} more cases")
    
    # Save detailed results
    res_path = os.path.dirname(csv_file_path)
    results_df.to_csv(f'{res_path}/csv_vs_images_analysis_{axis}.csv', index=False)
    print(f"\n Detailed results saved to: csv_vs_images_analysis_{axis}.csv\n")
    
    return results_df

def compute_diversity_score(filename, res_path, axis, nota=False):
    distributions = create_distributions(filename, nota=nota)

    entropies_as_dict = {}
    prompt_list=[]
    question_list =[]
    dist_list=[]
    entropy_list=[]
    norm_Dq_list=[]
    Dq_list = []
    maxD_list = []
    norm_entropy_list = []


    for key, dist in distributions.items():
        if not nota and 'none of the above' in dist:
        # Remove NoTA option since it's not being considered
            if dist['none of the above']==0:
                del(dist['none of the above'])
            else:
                print('ERROR!! NoTA value is non-zero!!')
                print(dist)
                break
        
        
        probabilities = [count for count in dist.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)  # Avoid log(0) error
        D_q = 2**(entropy)  # Exponential of entropy, hill number
        
        num_answers = len(dist)


        if num_answers >= 1:  # Avoid log(0) error when there's only one type of answer
            max_possible_D = num_answers
            normalized_D = (D_q-1) / (max_possible_D-1)  # Normalized to [0, 1] range
            max_possible_entropy = math.log2(max_possible_D)
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_D = 0
            normalized_entropy = 0
        
        prompt_list.append(key.split('_')[0])
        question_list.append(key.split('_')[1])
        dist_list.append(dist)
        entropy_list.append(entropy)
        norm_entropy_list.append(normalized_entropy)
        maxD_list.append(max_possible_D)
        Dq_list.append(D_q)
        norm_Dq_list.append(normalized_D)
        
        if not os.path.exists(res_path):
            os.makedirs(res_path)


        df2save = pd.DataFrame(
                        {'prompt': prompt_list,
                        'question': question_list,
                        'distribution': dist_list,
                        'entropy': entropy_list,
                        'normalised entropy':norm_entropy_list,
                        'D_q': Dq_list,
                        'max D': maxD_list,
                        'normalised D_q':norm_Dq_list
                        })
        
        if nota:
            df2save.to_csv(res_path+f'/{axis}-prompt-dist-with-nota.csv', index=False)
        else:
            df2save.to_csv(res_path+f'/{axis}-prompt-dist.csv', index=False)
            
        entropies_as_dict[key] = normalized_D

    mean_normalized_Dq = sum(norm_Dq_list) / len(norm_Dq_list)
    return mean_normalized_Dq

def load_model_predictions(df, nota=False):

    initial_len = len(df)
    df = df[df['answer'] != "['parsing failed']"]
    final_len = len(df)
    print(f"Removed {initial_len - final_len} rows with 'parsing failed' answers ({(1 - round(final_len/initial_len, 2))*100}%).")
    if not nota:
        df = df[~df['answer'].str.contains(r"none of the above", case=False, na=False, regex=True)]
        final_len = len(df)
        print(f"Removed {initial_len - final_len} rows with 'none of the above' answers ({(1 - round(final_len/initial_len, 2))*100}%).")
    return df

def create_distributions(df, nota=False):
    df = load_model_predictions(df, nota=nota)
    out_of_scope_answers = {}
  
    def count_answers(group, group_keys):
        
        prompt_id, question_id = group_keys
        key = f"{prompt_id}_{question_id}"

        answer_counts = {}
        
        answer_list = group['answer_list'].to_list()
        answer_list = [item.lower() for sublist in answer_list for item in sublist]
        answer_list = list(set(answer_list))

        for answer in answer_list:
            answer_counts[answer] = 0

        # Count each individual answer from potentially list-type answers
        for raw_answer in group['answer']:
            try:
                answer_list = ast.literal_eval(raw_answer) if isinstance(raw_answer, str) else raw_answer
            except:
                answer_list = [raw_answer]

            if not isinstance(answer_list, list):
                answer_list = [answer_list]

            for answer in answer_list:
                if answer in answer_counts:
                    answer_counts[answer] += 1
                else:           
                    key = f"{group['prompt'].iloc[0]}_{group['question'].iloc[0]}"
                    if key in out_of_scope_answers:
                        out_of_scope_answers[key][1] += 1
                    else:
                        out_of_scope_answers[key] = [answer, 1]

        total = sum(answer_counts.values())
        for key, count in answer_counts.items():
            if total != 0:
                answer_counts[key] = count / total
            else:
                answer_counts[key] = 0
        return answer_counts
    
    # Choose the appropriate grouping columns
    grouping_columns = ['prompt', 'question']

    result = df.groupby(grouping_columns).apply(
        lambda group: count_answers(group, group.name),
        # include_groups=False
    )

    distributions = {f"{idx[0]}_{idx[1]}": counts for idx, counts in result.items()}
    return distributions

def save_plots_from_distributions(dataset_name, entity_name, res_path, axis):
    print(f"\nProcessing {dataset_name} - {entity_name}")

    # Load the full dataframe
    df = pd.read_csv(f'{res_path}/{axis}-prompt-dist-final.csv')
    plot_path = f'{res_path}/plots-{axis}/'
    os.makedirs(plot_path, exist_ok=True)
    
    prompt_list = df['prompt'].to_list()
    question_list = df['question'].to_list()
    title_list = [p+'_'+q for p,q in zip(prompt_list,question_list)]
    dist_list = df['distribution'].to_list()
    norm_entropy_list = df['normalised D_q'].to_list()
    
    dist_list = [eval(item) for item in dist_list]
    
    seen_q = []
    for title, dist, norm_entropy, question in zip(title_list, dist_list, norm_entropy_list, question_list):
        question = question.replace('/','_')
        title = title.replace('/','_')
        curr_path = os.path.join(plot_path, question.replace(' ','_'))
        if question not in seen_q:
            os.makedirs(curr_path, exist_ok=True)
            seen_q.append(question)
        
        
        # Sort attribute values alphabetically
        x_labels = sorted(dist.keys())
        y_values = [dist[label] for label in x_labels]

        # Map attribute values to 'a', 'b', 'c', ...
        alpha_labels = [chr(97 + i) for i in range(len(x_labels))]
        label_mapping = dict(zip(alpha_labels, x_labels))

        plt.figure(figsize=(9, 6))
        bars = plt.bar(alpha_labels, y_values, color="steelblue", edgecolor="black", width=0.8)

        # Add values on top of bars
        for i, val in enumerate(y_values):
            plt.text(i, val + 0.02, f"{val:.2f}", ha='center', fontsize=10)

        # Axis setup
        plt.xlabel("Answer Choices", fontsize=12)
        plt.ylabel("Probabilities", fontsize=12)
        plt.ylim(0, 1)

        region = title.split('_')[0].split('A photo of a ')[-1].split('in ')[-1]

        # Updated title format: question [entity, region]
        # Wrap question text if it's too long
        wrapped_question = "\n".join(textwrap.wrap(question, width=60))  # Adjust width as needed
        plt.title(f"{wrapped_question} [{region}]", fontsize=14, pad=20)

        # Gridlines for readability
        # plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Manual legend-style annotation inside plot (top-right)
        legend_text = f"Hill No.: {norm_entropy:.3f}\n\n" + "\n".join(
                        [f"{k} → {v}" for k, v in label_mapping.items()]
                    )
        plt.gca().text(
                0.02, 0.95, legend_text,  # x = 0.02 (left), y = 0.95 (top)
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',  # align text with the left edge of the box
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round, pad=0.5'),
                linespacing=1.4,
                fontfamily='monospace'
            )

        fig_name = f"{region}_{question.replace(' ', '_')[:50]}.png"
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(curr_path, fig_name), dpi=300, bbox_inches="tight")
        plt.close()
    return



def main(args):

    dataset = args.dataset_name 
    entity = args.entity_name 
    axis = args.axis 
    res_path = args.output_path
    
    if axis == 'entity':
        df_F = pd.read_csv(f'{res_path}/vqa_F_predictions.csv')
        df_NF = pd.read_csv(f'{res_path}/vqa_NF_predictions.csv')
        df = pd.concat([df_F, df_NF], ignore_index=True)
        df.to_csv(f'{res_path}/{axis}.csv', index=False)
    else:
        df = pd.read_csv(f'{res_path}/step4_predictions.csv')
        df.to_csv(f'{res_path}/{axis}.csv', index=False)
    
    df = pd.read_csv(f'{res_path}/{axis}.csv')
    # df = df[(df['prompt'].str.contains('Nigeria')) | (df['prompt'].str.contains('United Kingdom'))]
    df["answer_list"] = df['answer_list'].apply(lambda x: eval(x.strip().lower()))
    df['answer'] = df['answer'].astype(str)
    df['answer'] = df['answer'].apply(lambda x: x.strip().lower())
    print()
    print(len(df), "rows in the original dataframe.")
    print(f"Total unique prompt+question combinations: {len(df.drop_duplicates(subset=['prompt', 'question']))}")
    print()
    print("Calculating distribution of answers, leaving out instances of NoTA.")
    s=compute_diversity_score(df, res_path, axis, nota=False)
    print(entity, dataset)
    print('Diversity (mean hill no) without NoTA:', s)
    print()
    # print(dist)
    print("Calculating distribution of answers, keeping instances of NoTA.")
    s=compute_diversity_score(df, res_path, axis, nota=True)
    print(entity, dataset)
    print(len(df), 'Diversity (mean hill no) with NoTA:', s)
    print()
    

    df_dist_nota = pd.read_csv(res_path+f'/{axis}-prompt-dist-with-nota.csv')
    print(f"Total unique prompt+question combinations in df_dist_nota: {len(df_dist_nota.drop_duplicates(subset=['prompt', 'question']))}")
    df_dist = pd.read_csv(res_path+f'/{axis}-prompt-dist.csv')
    print(f"Total unique prompt+question combinations in df_dist: {len(df_dist.drop_duplicates(subset=['prompt', 'question']))}")
    
    # Step 0: Analyze CSV vs images to identify low coverage questions
    df_drop = analyze_csv_vs_images(res_path+f'/{axis}.csv', dataset, axis, show_low_coverage=False)
    
    # Step 1: Check coverage and drop low coverage questions
    print()
    print(f"\n=== 1. COVERAGE ===")
    print("Dropping low coverage Qs")
    df_drop = df_drop[df_drop['low_coverage']==False]
    keep_pairs = set(zip(df_drop['prompt'], df_drop['question']))
    df_filtered = df_dist[df_dist.apply(lambda row: (row['prompt'], row['question']) in keep_pairs, axis=1)]
    
    # Step 2: Analyze low entropy questions and report
    if axis=='entity':
        threshold_entropy = 0.2
    elif axis=='background':
        threshold_entropy = 0.1
    print(f"\n=== 2. LOW Entropy < {threshold_entropy} ===")
    print("Before filtering low entropy questions: Unique ", len(df_filtered['question'].unique()), df_filtered['question'].unique())
    print()
    unique_question = df_filtered['question'].unique()
    question = []
    D_q = []
    for q in unique_question:
        df_sub = df_filtered[df_filtered['question']==q]
        normalised_D_q = df_sub['normalised D_q'].to_list()
        mean_normalised_D_q = sum(normalised_D_q) / len(normalised_D_q)
        if mean_normalised_D_q < threshold_entropy:
            print(f"{q} {round(mean_normalised_D_q,3)}")
            # Dropping low entropy questions only for background axis
            if axis=='background':
                continue
        else:
            question.append(q)
            D_q.append(mean_normalised_D_q)
                
    df = pd.DataFrame({'Question':question, 'Norm_Hill_no':D_q}, columns=['Question','Norm_Hill_no'])
    df.to_csv(res_path+f'/question_wise_scores_{axis}.csv', index=False)
    # also remove these low entropy questions from the keep_pairs for the next steps
    keep_pairs = {(prompt, q) for prompt, q in keep_pairs if q in question}
    print("After filtering low entropy questions: Unique", len(df['Question'].unique()), df['Question'].unique())
    
    # Step 3: Check nota analysis
    print()
    print(f"\n=== 3. NOTA ANALYSIS ===")
    print()
    df_filtered = pd.read_csv(res_path+f'/question_wise_scores_{axis}.csv')
    print(f"Unique questions in question_wise_scores_{axis}:")
    print(df_filtered['Question'].unique())
    print()
    high_nota = []
    df = pd.read_csv(f'{res_path}/{axis}.csv')
    df["answer_list"] = df['answer_list'].apply(lambda x: eval(x.strip().lower()))
    df['answer'] = df['answer'].astype(str)
    df['answer'] = df['answer'].apply(lambda x: x.strip().lower())
    df_nota_qs = df[df['answer'].str.contains("none of the above", na=False)]
    df_nota_qs = df_nota_qs[df_nota_qs['question'].isin(df_filtered['Question'].unique())]
    l=list(df_nota_qs['question'].value_counts().items())
    for i in range(len(df_nota_qs['question'].value_counts())):
        df_total = df[df['question']==l[i][0]]
        print(f'Q.{i+1} {l[i][0]:<130} NoTA: {l[i][1]:<5} Total: {len(df_total):<5} Percentage of Images showing this attribute: {(1-(l[i][1]/len(df_total)))*100}')
        if ((l[i][1]/len(df_total)))*100 > 30:
            high_nota.append(l[i][0])
    print("High NoTA questions (>30%):", high_nota)
    print()

    df_filtered = df_dist[df_dist.apply(lambda row: (row['prompt'], row['question']) in keep_pairs, axis=1)]
    df_nota_filtered = df_dist_nota[df_dist_nota.apply(lambda row: (row['prompt'], row['question']) in keep_pairs, axis=1)]

    if len(high_nota)>0:
        print("Comparing normalised hill no change after adding NoTA option for high NoTA questions:")
        print('----------------------------------------------------------------')
        for q in high_nota:
            df_sub = df_nota_filtered[df_nota_filtered['question']==q]
            normalized_D_q = df_sub['normalised D_q'].to_list()
            mean_normalized_D_q = sum(normalized_D_q) / len(normalized_D_q)
            
            df_sub = df_filtered[df_filtered['question']==q]
            normalized_D_q = df_sub['normalised D_q'].to_list()
            mean_normalized_D_q_old = sum(normalized_D_q) / len(normalized_D_q)
            
            print(f"{q} {round(mean_normalized_D_q_old,3)} -> {round(mean_normalized_D_q,3)}")
        print('----------------------------------------------------------------')
        print()

    # if any combination needs nota added
    for question in high_nota:
        mask = df_filtered['question'] == question

        for idx in df_filtered[mask].index:
            prompt_val = df_filtered.loc[idx, 'prompt']
            nota_row = df_nota_filtered[(df_nota_filtered['prompt'] == prompt_val) & (df_nota_filtered['question'] == question)]
            if not nota_row.empty:
                df_filtered.loc[idx] = nota_row.iloc[0]
    df_filtered.to_csv(f'{res_path}/{axis}-prompt-dist-final.csv', index=False)

    if args.save_plots:
        save_plots_from_distributions(dataset_name=dataset, entity_name=entity, res_path=res_path, axis=axis)
    
    overall_scores = []

    df = pd.DataFrame()
    df_temp = pd.read_csv(f'{res_path}/{axis}-prompt-dist-final.csv')
    df_temp['dataset'] = dataset
    df_temp['entity'] = entity
    # Load the full dataframe
    df = pd.concat([df, df_temp], ignore_index=True)
    
    df = df.groupby(['prompt', 'dataset'])

    for name, group in df:
        # GeoDiv scores
        #Normalised D_q
        norm_Dq = group['normalised D_q'].mean() 
        
        overall_scores.append({
            'prompt': name[0],
            'entity': entity,
            'dataset': name[1],
            'Normalised D_q': norm_Dq,
        })
    print(f"\n=== Overall Scores for {dataset} - {entity} ===")
    overall_scores_df = pd.DataFrame(overall_scores)
    overall_scores_df.to_csv(f'{res_path}/hillno_scores_{axis}.csv', index=False)
    print("Saved to:", f'{res_path}/hillno_scores_{axis}.csv')
        
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
    parser.add_argument("--save_plots", action='store_true',
                        help="Whether to save distribution plots for each question.")
    parser.add_argument("--axis",
                        choices=['entity', 'background'],
                        default='entity',
                        help="Name of the VDI axis you want to assess.")

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path, args.vqa_model, args.dataset_name, args.entity_name)

    main(args)