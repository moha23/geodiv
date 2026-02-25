import argparse
import os
from vqa_vdi import extract_scores


def get_model_processor(args):
    # Load the model 
    
    # for qwen, run in conda activate geodiv-qwen
    if args.vqa_model == 'qwen':     
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
        pipeline = [model, processor]

    # for gemini, run in conda activate geodiv
    elif args.vqa_model=='flash':        
        model = None
        processor = None
        pipeline = [model, processor]
    
    
    elif args.vqa_model =='gpt':
        from openai import OpenAI
        from utils.utils import load_oai_key
        client = OpenAI(api_key=load_oai_key())    
        version = "gpt-4o"
        model = [client, version]
        processor=None
        pipeline = [model, processor]
    
    else:
        raise ValueError(
            "Invalid model name. Choose from 'flash', 'qwen', or 'gpt'."
            )

    return pipeline

def main(args):
    
    # Load the model and processor based on the selected model
    pipeline = get_model_processor(args)
    
    results_dir = args.output_path
    os.makedirs(results_dir, exist_ok=True)
    generated_images_path = os.path.join(args.gen_img_path, args.entity_name)
    input_file_path = os.path.join('data', f"{args.step_name}.csv")
    interm_dir = os.path.join('data', args.vqa_model, args.entity_name) # to save intermediate csvs
    os.makedirs(interm_dir, exist_ok=True)

    extract_scores(args.thinking_budget, args.batch, args.axis, args.proj_id, input_file_path, generated_images_path, results_dir, pipeline[0], pipeline[1], vqa_model=args.vqa_model, dataset_name=args.dataset_name, step_name=args.step_name, entity_name=args.entity_name, interm_dir=interm_dir)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run VQA.")
    parser.add_argument("--vqa_model",
                        choices=['qwen', 'flash', 'gpt'],
                        default='qwen',
                        help="VLM to use")
    parser.add_argument("--axis",
                        choices=['entity', 'background'],
                        default='entity',
                        help="Name of the VDI axis you want to assess.")
    parser.add_argument("--entity_name",
                        choices=['house', 'bag', 'backyard', 'car', 'cooking pot', 'plate of food', 'dog', 'storefront', 'chair', 'stove'],
                        default='house',
                        help="The entity to work on.")    
    parser.add_argument("--output_path", type=str, required=False, 
                        default="./results/", 
                        help="Path to the output directory to save the scores.")
    parser.add_argument("--gen_img_path", type=str, required=False,  
                        default="images/sd21/",
                        help="Path to the generated images to be scored.")
    parser.add_argument("--step_name",
                        type=str,
                        choices=['visibility', 'vqa_F', 'vqa_NF', 'step1', 'step2', 'step3', 'step4'],
                        default='visibility',
                        help="VQA step name.")
    parser.add_argument("--dataset_name", type=str, required=False, default="sd21",
                        choices=['geode', 'sd21', 'sd3m', 'flux1', 'sd35'],
                        help="Name of the image dataset to be used/stored.")
    parser.add_argument("--proj_id",
                        type=str,
                        help="GCloud Project ID.")
    parser.add_argument("--prompts_file_path",
                        type=str,
                        default='country_prompts',
                        help="Prompts file to be used.")
    parser.add_argument("--batch", 
                        type=int, 
                        choices=[0,1], 
                        default=1, 
                        help="GCloud batch processing.")
    parser.add_argument("--thinking_budget",
                        type=int,
                        default=-1, # dynamic
                        help="Thinking budget.")

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path, args.vqa_model, args.dataset_name, args.entity_name)

    main(args)
