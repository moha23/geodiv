import os
import argparse
from vqa_sevi import extract_scores



def get_model_processor(args):
    if args.vqa_model.lower() == "qwen":
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        torch.set_default_dtype(torch.float16)
        print('visible devices', torch.cuda.device_count())
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
        pipeline = [model, processor]
            
    elif args.vqa_model.lower() == 'flash':
        pipeline=None
        
    else:
        raise ValueError(
            "Invalid model name. Choose from 'flash' or 'qwen'."
            )
    
    return pipeline




def main(args):
    
    results_dir = args.output_path
    os.makedirs(results_dir, exist_ok=True)
    generated_images_path = os.path.join(args.gen_img_path, args.entity_name)
    prompts_file_path = os.path.join('data', f"{args.prompts_file_path}.csv") 
    interm_dir = os.path.join('data', args.vqa_model, args.entity_name) # to save intermediate csvs
    os.makedirs(interm_dir, exist_ok=True)   
        
    # Load the model and processor based on the selected model
    pipeline = get_model_processor(args)

    axis = args.axis.lower()

    extract_scores(args.batch, args.thinking_budget, axis, args.proj_id, prompts_file_path, pipeline, generated_images_path, results_dir, args.vqa_model, dataset_name=args.dataset_name, entity_name=args.entity_name, interm_dir=interm_dir)
        


if __name__ == "__main__":
    # Load the model and processor based on the selected model
    parser = argparse.ArgumentParser(description="Give score for generated images")
    parser.add_argument("--vqa_model", type=str, required=False, default="qwen", 
                        choices=["qwen", "flash"], help="VLM to use")
    parser.add_argument("--axis", type=str, required=False, default="affluence", 
                        choices=["affluence", "maintenance"], help="Name of the SEVI axis you want to assess.")
    parser.add_argument("--gen_img_path", type=str, required=False,  
                        default="images/sd21/",
                        help="Path to the generated images to be scored.")
    parser.add_argument("--output_path", type=str, required=False, 
                        default="./results/", 
                        help="Path to the output directory to save the scores.")
    parser.add_argument("--entity_name",
                        choices=['house', 'bag', 'backyard', 'car', 'cooking pot', 'plate of food', 'dog', 'storefront', 'chair', 'stove'],
                        default='house',
                        help="The entity to work on.")
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
    parser.add_argument("--thinking_budget",
                        type=int,
                        default=-1, # dynamic
                        help="Thinking budget.")
    parser.add_argument("--batch", type=int, choices=[0,1], default=1, help="GCloud batch processing.")
    
    args = parser.parse_args()
    
    args.output_path = os.path.join(args.output_path, args.vqa_model, args.dataset_name, args.entity_name)
    
    main(args)
    