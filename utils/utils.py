import os
import subprocess
from typing import Optional
import shlex
import json
import re
import pandas as pd
from typing import Tuple, Optional

def add_images_to_csv(axis, prompts_file_path, generated_images_path, vqa_model, dataset_name, entity_name, interm_dir):
    df = pd.read_csv(prompts_file_path)
    new_rows = []
    # Iterate over each row in the original DataFrame
    for _, row in df.iterrows():
        prompt_id = row['prompt_id']
        img_dirpath = os.path.join(generated_images_path, str(prompt_id))
        image_paths = load_paths_for_prompt_id(vqa_model, dataset_name, img_dirpath)

        if not image_paths:
            # print(f"Did not find images for prompt_id {prompt_id}")
            continue
        # For each image_path, create a new row in the new DataFrame
        for image_path in image_paths:
            new_row = row.to_dict()
            new_row['image_path'] = image_path
            new_rows.append(new_row)
    # Create a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)    

    # Useful if you choose to use batch-inference
    new_df['image_name'] = new_df['image_path'].apply(lambda x: x.split('/')[-1].replace(".png", ""))
    new_df['custom_id'] = new_df.apply(lambda x: f"{x['entity_id']}_{x['prompt_id']}_{x['image_name']}", axis=1)
    new_df.drop(columns=['image_name'], inplace=True)
    filename = os.path.basename(prompts_file_path)
    new_csv_path = interm_dir+'/'+filename.replace(".csv", '_'+axis+'_'+entity_name+'_'+dataset_name+"_with_images.csv")
    new_df.to_csv(new_csv_path, index=False)
    return new_csv_path

def load_paths_for_prompt_id(model_name, dataset_name, dirpath: str, max_seed: Optional[int] = None):
    # load all images from path:
    image_paths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if file.endswith('.png'):
                if max_seed and int(file.split('_')[0]) >= int(max_seed):
                    print(f"Reached max seed: {max_seed}. Stopping image loading now.")
                    break
                image_paths.append(os.path.join(root, file))
                
                # Change local paths to GCS paths for Gemini/Flash
                if (model_name == 'flash'): 
                    image_paths = [path.replace(f'images/{dataset_name}', f'gs://{dataset_name}_gcs') for path in image_paths]
                
    return image_paths

def convert_gcs_to_local(gcs_path, dataset_name):
    """Convert GCS path to local path based on the mapping (reverse of your original)"""
    return gcs_path.replace(f'gs://{dataset_name}_gcs', f'images/{dataset_name}')


def upload_with_subprocess(local_file, bucket, destination):
    """Upload file to GCS bucket using gsutil command line via Python"""
    
    gcs_uri = f"gs://{bucket}/{destination}"
    
    # Use shlex.quote to safely escape paths with spaces
    safe_gcs_uri = shlex.quote(gcs_uri)
    safe_local_path = shlex.quote(local_file)
    command = f"gsutil cp  {safe_local_path} {safe_gcs_uri}"
    # command = ["gsutil", "cp", safe_local_path, safe_gcs_uri]
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("Upload successful!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Upload failed: {e}")
        print(f"Error: {e.stderr}")
        return False
    
def download_from_gcs(source_path, local_destination):
    """Download file from GCS bucket using gsutil command line via Python"""
    
    gcs_prefix = f"{source_path}"
    result = subprocess.run(
                            ["gsutil", "ls", gcs_prefix],
                            check=True,
                            capture_output=True,
                            text=True
                        )
    folders = result.stdout.strip().split("\n")
    latest_folder = sorted(folders)[-1]   # assumes lexicographic order = time order

    # build the full path to predictions.jsonl
    gcs_uri = f"{latest_folder}predictions.jsonl"

    print(f"Downloading from {gcs_uri}")

    # gcs_uri = gcs_prefix+'/*/predictions.jsonl'
    
    # Use shlex.quote to safely escape paths with spaces
    safe_gcs_uri = shlex.quote(gcs_uri)
    safe_local_path = shlex.quote(local_destination)
    # command = f'gsutil cp "gs://{bucket}/{source_path}" "{local_destination}"'
    command = f"gsutil cp {safe_gcs_uri} {safe_local_path}"
    # command = ["gsutil", "cp", safe_gcs_uri, safe_local_path]
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"Download successful!")
        print(f"Downloaded: {safe_gcs_uri} → {safe_local_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print(f"Error: {e.stderr}")
        return False

def quick_response_check(jsonl_file):
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            response = data.get('response', {})
            candidates = response.get('candidates', [])
            
            if candidates and candidates[0].get('content', {}).get('parts'):
                text = candidates[0]['content']['parts'][0].get('text', '')
                print(f"Line {i+1}: {text[:50]}...")
            else:
                print(f"Line {i+1}: No valid response")
            
            if i >= 4:  # Show first 5 only
                break
            
def read_jsonl(df, jsonl_file):
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            custom_id = data.get('key', None)
            try:
                candidates = data["response"]["candidates"]
            except Exception as e:
                # print("Skipping for custom_id", custom_id)
                df.loc[df['custom_id'] == custom_id, 'reasoning_steps'] = "Did not run"
                df.loc[df['custom_id'] == custom_id, 'answer'] = "Did not run"
                continue
            thoughts_text = ""
            answer_text = ""
            for candidate in candidates:
                parts = candidate["content"]["parts"]
                for part in parts:
                    if part.get("thought", False):  # This is thinking text
                        thoughts_text = part["text"]
                    else:  # This is the actual response
                        answer_text = part["text"]
            thoughts_text = ' '.join(thoughts_text.splitlines())
            thoughts_text = ' '.join(thoughts_text.split())


            df.loc[df['custom_id'] == custom_id, 'reasoning_steps'] = str(thoughts_text)
            df.loc[df['custom_id'] == custom_id, 'answer'] = str(answer_text)
    return df

def get_clean_response(content, string=False):
    
    print("\n=======================")
    print(content)
    # Ensure the response does NOT contain markdown-style JSON formatting
    content_clean = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if content_clean:
        content_clean = content_clean.group(1).strip()
        try:
            content_clean = json.loads(content_clean)
            print("Parsed JSON successfully")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            try:
                # This is useful for step 4 VQA of background axis in VDI
                reasoning, answer = extract_multiline_blocks(content_clean)
                print("Successfully read multiline")
                print("\nReasoning Steps:", reasoning)
                print("Final Answer:", answer)
                return reasoning, answer
            except Exception as e:
                print(f"Further Error extracting multiline blocks: {e}")
                content_clean = {"reasoning": [], "answer": "Parsing failed"}

    else:      
        content_clean = content.lower().replace("answer:", "").strip()  # fallback to cleaning the raw response if no markdown formatting is found
        if string:
            content_clean = content_clean
        else:
            content_clean = int(content_clean) if content_clean.isdigit() else content_clean
    # Extract final values
    try:
        reasoning = content_clean.get("reasoning", [])
        answer = content_clean.get("answer", "No answer found") 
    except Exception as e:
        print(f"Error extracting reasoning steps and answer: {e}")
        reasoning = []
        if string:
            answer = content_clean
        else:
            answer = content_clean if isinstance(content_clean, int) else "No answer found"

    print("\nReasoning Steps:", reasoning)
    print("Final Answer:", answer)
    return reasoning, answer


def cleanup(x):
    x = x.replace("```json","").replace("```","").strip()
    return x

def clean_dataframe(df):
    # This function normalizes messy answer formats (like "answer: 3", {"score": 4}, etc.) into clean numeric strings (1-5)
    
    df['answer'] = df['answer'].astype(str)
    df['answer'] = df['answer'].apply(lambda x: cleanup(x))

    for _, row in df.iterrows():
        if row['answer'] not in ['1', '2', '3', '4', '5', '1.0', '2.0', '3.0', '4.0', '5.0']:
            # First cleanup attempt: remove "answer: " prefix and whitespace
            ans = row['answer']
            ans = ans.replace('answer: ', '').strip()
            df.loc[df['image_path'] == row['image_path'], 'answer'] = ans
            
            # Second cleanup attempt: extract number from JSON-like strings
            ans = row['answer']
            # Try to find pattern like "score": 3
            new_ans = re.search(r'"score":\s*(\d+)', ans)
            if not new_ans:
                # If not found, try pattern like "answer": 3
                new_ans = re.search(r'"answer":\s*(\d+)', ans)
            
            if new_ans:
                ans = new_ans.group(1)
            else:
                ans = ans.replace("answer:","").strip()

            df.loc[df['image_path'] == row['image_path'], 'answer'] = ans
    return df
        
def load_oai_key():
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    file_path = 'oai_key.txt'
    with open(file_path, 'r') as file:
        key = file.read().strip() 
    return key 

def extract_multiline_blocks(blob: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract multi-line substrings following 'reasoning:' and 'answer:'.
        - reasoning: from end of its label line up to the next label ('answer:') or EOF.
        - answer: from end of its label line to EOF.
        Returns (reasoning_block, answer_block) as raw strings (preserve newlines/indent).
        """
        # print("Inside the func, got:", blob)
        text = blob.replace('\r\n', '\n').replace('\r', '\n')

        # Label lines: allow leading spaces; label followed by ':' possibly with trailing content
        rs_line = re.compile(r'(?im)^(\s*reasoning\s*:\s*)(.*)$')
        an_line = re.compile(r'(?im)^(\s*answer\s*:\s*)(.*)$')

        rs = rs_line.search(text)
        an = an_line.search(text)

        reasoning_block = None
        answer_block = None

        if rs:
            # Start right after the label colon; include any same-line tail (group 2)
            rs_start = rs.end(1)
            # If same-line tail exists, take it plus newline and then continue
            # But we want the block from the rest of this line and subsequent lines until next label.
            # Compute end at next label line (answer) if it appears after rs_start, else EOF.
            end = an.start(1) if (an and an.start(1) > rs_start) else len(text)
            reasoning_block = text[rs_start:end].lstrip('\n').rstrip()  
            reasoning_block = ', '.join(line.strip() for line in reasoning_block.replace('-', '').split('\n') if line.strip())

        if an:
            an_start = an.end(1)
            answer_block = text[an_start:].lstrip('\n').rstrip()

        return reasoning_block, answer_block