import math
import os
import re
import ast
import argparse
import textwrap
import matplotlib.pyplot as plt
import pandas as pd


def compute_diversity_score(filename, res_path, axis):
    distributions = create_distributions(filename)

    entropies_as_dict = {}
    prompt_list=[]
    question_list =[]
    dist_list=[]
    entropy_list=[]
    norm_Dq_list=[]
    Dq_list = []
    maxD_list = []
    norm_entropylist = []


    for key, dist in distributions.items():
        
        
        probabilities = [count for count in dist.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)  # Avoid log(0) error
        D_q = 2**(entropy)  # Exponential of entropy, hill no
        
        num_answers = len(dist)

        if num_answers >= 1:  # Avoid log(0) error when there's only one type of answer
            max_possible_D = num_answers
            normalised_D = (D_q-1) / (max_possible_D-1)  # normalised to [0, 1] range
            max_possible_entropy = math.log2(max_possible_D)
            normalised_entropy = entropy / max_possible_entropy
        else:
            normalised_D = 0
            normalised_entropy = 0
        
        prompt_list.append(key.split('_')[0])
        question_list.append(key.split('_')[1])
        dist_list.append(dist)
        entropy_list.append(entropy)
        norm_entropylist.append(normalised_entropy)
        maxD_list.append(max_possible_D)
        Dq_list.append(D_q)
        norm_Dq_list.append(normalised_D)
        
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        
        df2save = pd.DataFrame(
                        {'prompt': prompt_list,
                        'question': question_list,
                        'distribution': dist_list,
                        'entropy': entropy_list,
                        'normalised entropy':norm_entropylist,
                        'D_q': Dq_list,
                        'max D': maxD_list,
                        'normalised D_q':norm_Dq_list
                        })
        df2save.to_csv(res_path+f'/{axis}-prompt-dist-final.csv', index=False)
          
        entropies_as_dict[key] = normalised_D

    mean_normalised_D_q = sum(norm_Dq_list) / len(norm_Dq_list)
    return mean_normalised_D_q

def load_model_predictions(df):

    initial_len = len(df)
    df = df[df['answer'] != "['parsing failed']"]
    df = df[df['answer'] != "Did not run"]
    final_len = len(df)
    print(f"Removed {initial_len - final_len} rows with 'parsing failed' or 'Did not run' answers ({(1 - round(final_len/initial_len, 2))*100}%).")
    
    return df

def create_distributions(df):
    df = load_model_predictions(df)
    out_of_scope_answers = {}

    def count_answers(group, group_keys):
        prompt_id, question_id = group_keys
        key = f"{prompt_id}_{question_id}"

        answer_counts = {}

        answer_list = group['answer_list'].to_list()
        answer_list = [item for sublist in answer_list for item in sublist]
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
        lambda group: count_answers(group, group.name)
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

countries = ['United Kingdom', 'Nigeria', 'Turkey', 'Indonesia', 'Colombia', 'Japan', 'India', 'China',
             'United States of America', 'Mexico', 'United Arab Emirates', 'South Korea',
             'Spain', 'Italy', 'Egypt', 'Philippines']

def main(args):

    dataset = args.dataset_name 
    entity = args.entity_name 
    axis = args.axis 
    res_path = args.output_path
    axis = args.axis

    df = pd.read_csv(f'{res_path}/{axis}_predictions.csv')
    df["answer_list"] = "[1,2,3,4,5]"
    df["question"] = f"What's the {axis} score?"
    df['question_id']='s0'
    print(f"{res_path}/{axis}_predictions.csv" )
    df["answer_list"] = df['answer_list'].apply(lambda x: eval(x.strip().lower()))

    print("Calculating distribution of answers")
    s=compute_diversity_score(df, res_path, axis)
    print(entity, dataset)
    print(len(df), 'Diversity (mean hill no):', s)
    print()
    
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
    parser.add_argument("--axis", type=str, required=False, default="affluence", 
                        choices=["affluence", "maintenance"], help="Name of the SEVI axis you want to assess.")

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path, args.vqa_model, args.dataset_name, args.entity_name)

    main(args)