import json
import os
import base64
import time
import pandas as pd
from datetime import datetime
from utils.utils import upload_with_subprocess, download_from_gcs, quick_response_check, read_jsonl, get_clean_response, clean_dataframe, add_images_to_csv

def extract_scores(batch, thinking_budget, axis, proj_id, prompts_file_path, pipeline, generated_images_path, results_dir, vqa_model, dataset_name, entity_name, interm_dir):
    
    new_csv_path = add_images_to_csv(axis, prompts_file_path, generated_images_path, vqa_model, dataset_name, entity_name, interm_dir)
    df = pd.read_csv(new_csv_path)
    print("Total length:", len(df))
    
    if batch==1 and vqa_model == 'flash': # batch inference only for gemini/flash
        from google import genai
        from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
        
        model = "gemini-2.5-flash"
        jsonl_path = create_jsonl_file(new_csv_path, axis, entity_name, thinking_budget)
        filename = os.path.basename(jsonl_path)
        upload = upload_with_subprocess(jsonl_path, "geodiv-batch-results", f"{axis}/{vqa_model}/{filename}")
        if not upload:
            print("Stopping since file upload failed")
            return

        client = genai.Client(vertexai=True,
                            project=proj_id,
                            location="us-central1",
                            http_options=HttpOptions(api_version="v1"))

        output_uri = f"gs://geodiv-batch-results/{axis}/{vqa_model}/{axis}_{entity_name}_{dataset_name}_results"
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
            download_from_gcs(output_uri, results_dir+f'/{axis}_predictions.jsonl')
            print()
            print('Checking if the saved file has response field:')
            quick_response_check(results_dir+f'/{axis}_predictions.jsonl')
            df['img_name'] = df['image_path'].apply(lambda x: os.path.basename(x).split('.png')[0])
            df['custom_id'] = df.apply(
                lambda row: f"{row['entity_id']}_{row['prompt_id']}_{row['img_name']}", axis=1
            )
            df = read_jsonl(df, results_dir+f'/{axis}_predictions.jsonl')
            df.sort_values(['entity_id', 'prompt_id'], inplace=True)        
            df = clean_dataframe(df)
            df.to_csv(os.path.join(results_dir, f"{axis}_predictions.csv"), index=False)
            
    else:
        
        new_rows = []
        for _, row in df.iterrows():
            if vqa_model == 'qwen':
                encoded_img = encode_image(row["image_path"]) 
                encoded_img = f"data:image/jpeg;base64,{encoded_img}"
            else:
                encoded_img = row["image_path"]

            start_time = time.time()
            reasoning, answer = single_call(proj_id, pipeline, encoded_img, vqa_model, axis, entity_name)
            
            end_time = time.time()
            row["reasoning"] = reasoning
            row["answer"] = answer
            row["time_taken"] = end_time - start_time
            new_rows.append(row)

            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(os.path.join(results_dir, f"{axis}_predictions.csv"), index=False)
            
    
def create_jsonl_file(csv_path, axis, entity_name, thinking_budget):

    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    filename = filename.replace(".csv", ".jsonl")
    folder = os.path.dirname(csv_path)
    
    if axis == 'affluence':
        with open('./prompts/affluence.txt', 'r') as file:
            system_instructions= file.read()
        instructions= f""" 
                    What is the socio-economic character of the image based on visual cues alone?
                    Respond only with a single integer between 1 (impoverished) and 5 (luxurious).
                    Selection:
        """
                    
    elif axis=='maintenance':
        
        with open('./prompts/maintenance.txt', 'r') as file:
            system_instructions= file.read()
        instructions= f"""
                    Dominant entity: {entity_name}
                    What is the physical condition of the specified dominant entity in the image, based on visual cues alone?
                    Respond only with a single integer between 1 (severe damage) and 5 (excellent condition).
                    
                    Selection:
        """
        
    with open(folder+'/'+filename, "w") as f:
        for _, row in df.iterrows():
            
            request_obj = {
                "key": f"{row['custom_id']}",                    
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

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def single_call(proj_id, model, image_url, model_name, axis, entity_name):
    
    if axis == 'affluence':
        with open('./prompts/affluence_with_reason.txt', 'r') as file:
            system_instructions= file.read()

        instructions = f"""
            What is the socio-economic character of the image based on visual cues alone?
            Respond only with a single integer between 1 (impoverished) and 5 (luxurious), and provide the reasoning.
            Selection:
            """
    elif axis == 'maintenance':
        with open('./prompts/maintenance_with_reason.txt', 'r') as file:
            system_instructions= file.read()

        instructions = f"""
            Dominant entity: {entity_name}
            What is the physical condition of the specified dominant entity in the image, based on visual cues alone?
            Respond only with a single integer between 1 (severe damage) and 5 (excellent condition), and provide the reasoning.
            Selection:
            """

    if model_name == 'flash':
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
        
        # Initialize Vertex AI
        PROJECT_ID = proj_id
        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-2.5-flash",
                                system_instruction=[system_instructions])
        
        image_file = Part.from_uri(
            image_url, mime_type="image/png"
        )
        
        max_retries = 10
        retry_count = 0
        content = None
        
        while retry_count < max_retries: 
            try:
                response = model.generate_content([image_file,instructions],generation_config={
                                    "temperature": 0.0,  # Deterministic
                                    "top_p":0.01,
                                    "top_k":1,
                                    "candidate_count":1,
                                    "max_output_tokens": 4000}  # Limit response length
                                    )
                content = response.text
                if content:
                    break  # Exit loop if response is successful
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Attempt {retry_count} failed: {e}. Retrying after 10 seconds...")
                    time.sleep(10)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    content = "Error: Max retries exceeded"
                
    elif model_name == 'qwen':
        processor = model[1]
        model = model[0]
        conversation = [{"role":"user",
                        "content":[{"type":"image","image": image_url},
                                    {"type":"text","text":system_instructions+'\n'+instructions}]}]
        inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)
        # Inference: Generation of the output
        output_ids = model.generate(**inputs, max_new_tokens=4000)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        content = output_text[0]
        
    return get_clean_response(content)