import os 
import time
import json
import base64
import pandas as pd
from datetime import datetime
from utils.utils import upload_with_subprocess, download_from_gcs, quick_response_check, read_jsonl, get_clean_response, add_images_to_csv

def extract_scores(thinking_budget, batch, axis, proj_id, input_file_path, generated_images_path, results_dir, model, processor, vqa_model, dataset_name, step_name, entity_name, interm_dir):
    
    if (step_name not in ['vqa_NF','step2', 'step3', 'step4']):
        new_csv_path = add_images_to_csv(axis, input_file_path, generated_images_path, vqa_model, dataset_name, entity_name, interm_dir)
    else:
        filename = os.path.basename(input_file_path)
        new_csv_path = interm_dir+'/'+filename.replace(".csv", '_'+axis+'_'+entity_name+'_'+dataset_name+"_with_images.csv")
        
    df = pd.read_csv(new_csv_path)
    print("Total length:", len(df))
    
    if step_name in ['visibility', 'vqa_NF']:
        print('Working only on NF questions [multiple choice possible, non determinate]. Total count:', len(df))
    elif step_name == 'vqa_F':
        print('Working only on F questions [determinate]. Total count:', len(df))
    else:
        print(f'Working on {step_name} questions of background axis. Total count: {len(df)}')
    
    # add "none of the above" to the answer list
    df["answer_list"] = df['answer_list'].apply(lambda x: eval(x.strip().lower()))
    if step_name not in ['visibility', 'step1', 'step2', 'step3']:
        df['answer_list'] = df['answer_list'].apply(lambda x: x + ["none of the above"]) 
        
    if batch==1 and vqa_model == 'flash': # batch inference only for gemini/flash:
        from google import genai
        from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
        
        model = "gemini-2.5-flash"
        
        if step_name in ['visibility', 'vqa_F']:
            choice = "single"
        elif step_name in ['vqa_NF']:
            choice = "multiple"
        else:
            choice = step_name
        
        jsonl_path = create_jsonl_file(df, new_csv_path, choice, entity_name, thinking_budget)
        filename = os.path.basename(jsonl_path)
        upload = upload_with_subprocess(jsonl_path, "geodiv-batch-results", f"{axis}/{vqa_model}/{filename}")
        if not upload:
            print("Stopping since file upload failed")
            return

        client = genai.Client(vertexai=True,
                            project=proj_id,
                            location="us-central1",
                            http_options=HttpOptions(api_version="v1"))
        output_uri = f"gs://geodiv-batch-results/{axis}/{vqa_model}/{step_name}_{entity_name}_{dataset_name}_results"
        # See the documentation: https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.create
        
        job = client.batches.create(
            # To use a tuned model, set the model param to your tuned model using the following format:
            # model="projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}
            model=model,
            # Source link: https://storage.cloud.google.com/cloud-samples-data/batch/prompt_for_batch_gemini_predict.jsonl
            src=f"gs://geodiv-batch-results/{axis}/{vqa_model}/{filename}",
            config=CreateBatchJobConfig(dest=output_uri),
        )
        print(f"Job name: {job.name}")
        print(f"Job state: {job.state}")
        # See the documentation: https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob
        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        print(f"Monitoring job: {job.name}")
        start_time = datetime.now()
        check_interval = 100 #seconds
        while True:
            try:
                job = client.batches.get(name=job.name)
                elapsed = datetime.now() - start_time
                
                print(f"[{elapsed}] Job state: {job.state}")
                
                if job.state in completed_states:
                    print(f"Job completed with state: {job.state}")
                    break
                    
                time.sleep(check_interval) 
                
            except Exception as e:
                print(f"Error checking job status: {e}")
                time.sleep(check_interval)
                continue
        
        
        if job.state.name == 'JOB_STATE_FAILED':
            if job.error:
                print(f"Error: {job.error}")
            else:
                print("Job failed but no error details available")
                
            
        if job.state == "JOB_STATE_SUCCEEDED":
            download_from_gcs(output_uri, results_dir+f'/{step_name}_predictions.jsonl')
            print()
            print('Checking if the saved file has response field:')
            quick_response_check(results_dir+f'/{step_name}_predictions.jsonl')
            
            df['img_name'] = df['image_path'].apply(lambda x: os.path.basename(x).split('.png')[0])
            df['custom_id'] = df.apply(
                lambda row: f"{row['entity_id']}_{row['prompt_id']}_{row['question_id']}_{row['img_name']}", axis=1
            )
            df = read_jsonl(df, results_dir+f'/{step_name}_predictions.jsonl')
            df.sort_values(['entity_id', 'prompt_id'], inplace=True)     
            df.to_csv(results_dir+f'/{step_name}_predictions.csv', index=False)
            print('Saved CSV from JSON')
            
    else:

        new_rows = []
        for _, row in df.iterrows():
            question = row["question"]
            answer_list = row["answer_list"]
            if vqa_model in ['qwen', 'gpt']:
                image_url = row["image_path"]
                encoded_img = encode_image(image_url)
                encoded_img = f"data:image/jpeg;base64,{encoded_img}"
            else:
                encoded_img = row["image_path"]

            start_time = time.time()
            if step_name in ['visibility', 'vqa_F', 'step1', 'step2', 'step3']:
                choice = 'single'
            elif step_name in ['vqa_NF']:
                choice = 'multiple'
            else:
                choice = step_name
            reasoning_steps, answer = single_call(proj_id, choice, entity_name, question, answer_list, encoded_img, vqa_model, model, processor)
            end_time = time.time()
            
            row["time_taken"] = end_time - start_time
            row["reasoning_steps"] = reasoning_steps
            row["answer"] = answer
            new_rows.append(row)

            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(os.path.join(results_dir, f"{step_name}_predictions.csv"), index=False)
            
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_jsonl_file(df, csv_path, choice, entity_name, thinking_budget):
    
    filename = os.path.basename(csv_path)
    filename = filename.replace(".csv", ".jsonl")
    folder = os.path.dirname(csv_path)
    

    with open(f'./prompts/{choice}.txt', 'r') as file:
            system_instructions = file.read()
    
    
    with open(folder+'/'+filename, "w") as f:
        for _, row in df.iterrows():
            question = row["question"]
            answer_list = row["answer_list"]
            img_name = os.path.basename(row['image_path']).split('.png')[0]
            custom_id = f"{row['entity_id']}_{row['prompt_id']}_{row['question_id']}_{img_name}"
            
            if choice == 'step1':
                instructions = f"""
                    Entity: {entity_name}
                    Question: Is there any visible background in the image apart from the {entity_name}?
                    Categories: {answer_list}
                    Selection:
                    """
            else:
                instructions = f""" 
                    Entity: {entity_name}
                    Question: {question}
                    Categories:{answer_list}
                    answer:
                    """
            
            request_obj = {
                "key": f"{custom_id}",                    
                "request": {
                            "contents": [{
                                "role": "user",
                                "parts": [
                                        {"text":instructions},
                                        {"file_data": {"file_uri": row['image_path'], "mime_type": "image/png"}}
                                    ]
                                }],
                            "system_instruction": {
                                "parts": [       
                                        {"text":system_instructions}
                                ]
                            },
                            "generation_config": {
                                "temperature": 0.0,
                                "topP": 0.01,
                                "topK": 1,
                                "candidateCount": 1,
                                "maxOutputTokens": 4000,
                                "thinkingConfig": {
                                                    "includeThoughts": True,
                                                    "thinkingBudget": thinking_budget
                                                }
                                }
                            }}
            f.write(json.dumps(request_obj) + "\n")

    return folder+'/'+filename


def single_call(proj_id, choice, entity_name, question, answer_list, image_url, vqa_model, model, processor):
    
    with open(f'./prompts/{choice}_with_reason.txt', 'r') as file:
        system_instructions = file.read()

    if choice == 'step1':
        instructions = f"""
            Entity: {entity_name}
            Question: Is there any visible background in the image apart from the {entity_name}?
            Categories: {answer_list}
            Selection:
            """
    else:
        instructions = f""" 
            Entity: {entity_name}
            Question: {question}
            Categories:{answer_list}
            answer:
            """

    if vqa_model=='qwen':
        conversation = [{"role":"user",
                        "content":[{"type":"image", "image":image_url},
                                   {"type":"text", "text":system_instructions+'\n'+instructions}]
                        }]

        inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)

        # Inference: Generation of the output
        output_ids = model.generate(**inputs, max_new_tokens=2000, temperature=0.01)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        content = output_text[0]

    elif vqa_model == 'flash':
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
        # Initialize Vertex AI
        PROJECT_ID = proj_id
        vertexai.init(project=PROJECT_ID, location="us-east5")
        model = GenerativeModel("gemini-2.5-flash",
                                system_instruction=[system_instructions])
        
        image_file = Part.from_uri(image_url, mime_type="image/png")
        
        count=0
        while count<10:
            try:
                response = model.generate_content([image_file,instructions],generation_config={
                                                        "temperature": 0.0,  # Deterministic
                                                        "top_p":0.01,
                                                        "top_k":1,
                                                        "candidate_count":1,
                                                        "max_output_tokens": 4000}  # Limit response length
                                                        )
                content = response.text
                break
            except Exception as e:
                print(f"Busy or something: {e}. Retrying after 20 seconds...")
                time.sleep(20)
                
            count+=1

    return get_clean_response(content, string=True)





