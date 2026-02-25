import re
import json
import os
import argparse
import pandas as pd


def generate_answer_lists(model, processor, dataset_path, model_name):
    
    prompts_df = pd.read_csv("/entity_prompts.csv")
    questions_df = pd.read_csv("/final_questions.csv")

    df = prompts_df.merge(questions_df, on=['entity', 'entity_id'])

    captions = df['prompt'].tolist()
    questions = df['question'].tolist()
    caption_ids = df['prompt_id'].tolist()
    question_ids = df['question_id'].tolist()
    entity_ids = df['entity_id'].tolist()
    entity_list = df['entity'].tolist()
    regions = df['region'].to_list()

    entities_answer_list = []
    seen_ids = set()
    for caption, question, caption_id, q_id, entity, c_id, region in zip(captions, questions, caption_ids, question_ids, entity_list, entity_ids, regions):
            
        curr_id = f"{c_id}_{q_id}"
        if curr_id in seen_ids:
            continue
        instruction1 = f"""
                        I have a question that is asked about an image. I will provide you with the question and a caption of the image. Your job is to first carefully read the question and analyze, then hypothesize plausible answers to the question assuming you could examine the image (instead, you examine the caption). 
                        The answers should be in a list, as in the example below. 
                        Do not write anything other than the list of plausible answers.
                        Do not provide extra details to your answers in parentheses (e.g., white and NOT 'white (for decorated cookies)').
                        Do your best to be succinct and not overly-specific.
                        If the question is very open-ended, like 'Is there anything on the table?' or 'Is the cake decorated with any specific theme or design?', the answer list should be strictly ['yes', 'no'].
                        
                        Example:
                        Caption: a helmet in a bike shop
                        Question: What type of helmet is depicted in the image?  
                        Plausible answer list: ["motorcycle helmets",
                                    "bicycle helmets",
                                    "football helmets",
                                    "construction helmets",
                                    "military helmets",
                                    "firefighter helmets",
                                    "rock climbing helmets",
                                    "hockey helmets"]
                        """
        instruction = f"""
                        Caption: {caption}
                        Question: {question} 
                        Plausible answer list:
                    """
        
        if model_name=='qwen':
            conversation = [{
                "role":"user", 
                "content":[{"type":"text","text":instruction1+instruction}]
                }]

            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)


            # Inference: Generation of the output
            output_ids = model.generate(**inputs, max_new_tokens=1000, temperature=0.01)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            answer_list = output_text[0]

        elif model_name=='gemini':
                import vertexai
                from vertexai.generative_models import GenerativeModel, Part
                # Initialize Vertex AI
                PROJECT_ID = "" # add proj id
                vertexai.init(project=PROJECT_ID, location="europe-west8")
                model = GenerativeModel(model_name="gemini-2.5-pro",
                                        system_instruction=[instruction1]) 
                response = model.generate_content([instruction], generation_config={
                                                    "temperature": 0.0,  # Deterministic
                                                    "top_p":0.01,
                                                    "top_k":1,
                                                    "candidate_count":1,
                                                    "max_output_tokens": 4000}  # Limit response length
                                                    )
                answer_list = response.text
                    
        elif model_name=='gpt':
            client = model[0]
            version = model[1]
            response = client.chat.completions.create(
            model=version,
                messages=[{"role": "user", "content": [{"type": "text", "text": instruction}]}],
                temperature=0.0
            )
            answer_list = response.choices[0].message.content
       
        print(answer_list)
        seen_ids.add(curr_id)

        # add answer_list
        entities_answer_list.append({
            'entity_id': c_id,
            'entity': entity,
            'prompt_id': caption_id,
            'prompt': caption,
            'question_id': q_id,
            'question': question,
            'answer_list': answer_list,
            'region': region
        })

        entities_answer_list_df = pd.DataFrame(entities_answer_list)
        entities_answer_list_df.to_csv(dataset_path + "/unfiltered_answers.csv", index=False)


def filter_answer_lists(model, processor, dataset_path, model_name, paradigm, isqwen):

    c=0
    
    df = pd.read_csv(dataset_path + "/unfiltered_answers.csv") 

    df['answer_list'] = df['answer_list'].apply(eval)
    
    answer_list_dict = {}  # Initialize an empty dictionary to store results


    for (entity, question), group in df.groupby(['entity', 'question']):
        answer_list = group['answer_list'].tolist()        
        answer_list = [item.lower() for sublist in answer_list for item in sublist]
        answer_list = list(set(answer_list))

        answer_list_dict[(entity, question)] = answer_list
    
        
    filtered_answer_list = {}
    reasoning = {}
    for key, answer_list in answer_list_dict.items():
        entity_or_prompt, question = key
        

        instruction1 = f"""
                You are provided with a entity, a question about an image of this entity, and a list of possible answers. 
                Your task is to filter out answers that do not belong in the final list based on the following five filtering criteria:  
                
                (1) Out of Scope -- If an answer belongs to a completely different category than the rest, remove it. Example: If all answers describe number of table legs, but one says "wooden surface", remove it. 
                (2) “None of the Above” -- Do not allow answers that suggest no correct answer exists, such as "none", "no visible toppings", etc. Remove these.
                (3) Semantic Redundancy -- If two answers mean the same thing but one is more specific, keep the broader term and remove the more specific one. Example: Keep "chocolate" and remove "chocolate drizzle". 
                (4) Difficult to Detect from an Image -- If an answer cannot be determined by just looking at the image, remove it. 
                (5) Difficult to Distinguish from an Image -- if it is possible to visually detect but difficult to distinguish between two answers, either keep the most visually recognizable one or replace both answers with a new broader category.
                
                However, if an answer list consists of only ['yes','no'], it does not require filtering.
                
                How to Respond: First, carefully read the entity, question and answers. Then, apply each filtering rule and explain which answers are removed and why. Finally, provide the reasoning and the filtered answers list obtained by taking into account the reasoning steps. Provide the response in JSON format with the following structure:

                "reasoning_steps": ["Step 1", "Step 2", ...],
                "filtered_answers": ["answer1", "answer2", "answer3"]
                
                Example 1

                    Entity: A photo of Popcorn
                    Question: Are there any visible toppings or additions, such as butter or cheese?
                    Answers: ["no", "yes", "chocolate", "cinnamon", "butter", "none", "chocolate drizzle", "no visible toppings", "plain", "caramel", "cheese", "herbs", "truffle oil"]
                    reasoning_steps: [""no" and "yes" -- Out of scope, as they do not describe specific toppings whereas the other answers do (Criterion 1)", ""none" and "no visible toppings" -- Removed (Criterion 2: "None of the above")", ""chocolate drizzle" and "chocolate" -- "chocolate drizzle" is more specific, so remove it (Criterion 3: Redundancy)", "herbs" and "truffle oil" are too difficult to detect from image, so remove it (Criterion 4: Difficult to Detect from an Image)"]
                    filtered_answers: ['chocolate', 'cinnamon', 'butter', 'plain', 'caramel', 'cheese']

                Example 2

                    Entity: A photo of a table
                    Question: How many legs does the table have?
                    Answers:["no legs", "no", "yes", "one central pedestal", "one leg", "two trestle supports", "a trestle base", "two legs", "six legs", "a pedestal base", "three legs", "multiple legs", "five legs", "four legs"]
                    reasoning_steps: ["anything with 'trestle' is too specific and out of scope (criterion 1)", ""no leg"  Removed as it matches Criterion 2", ""two legs", "three legs", "four legs", "five legs", "six legs" -- Redundant. Keep the broadest term, "multiple legs", and remove the others (Criterion 3)"]
                    filtered_answers: ['one leg', 'multiple legs']
                
                """
        instruction = f"""
                        Entity: {entity_or_prompt}
                        Question: {question}
                        Answers: {answer_list}
                        Output:
                        """
        # Skip filtering if yes/no answer-lists
        if answer_list == "['yes', 'no']":
            filtered_answer_list[key] = answer_list
            reasoning[key] = ["No filtering needed for binary answers"]
        else:
            if model_name=='qwen':
                conversation = [{
                    "role":"user", 
                    "content":[{"type":"text", "text":instruction1+instruction}]
                    }]

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)


                # Inference: Generation of the output
                output_ids = model.generate(**inputs, max_new_tokens=1000, temperature=0.01)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
                content = output_text[0]
                
            elif model_name=='gemini':
                import vertexai
                from vertexai.generative_models import GenerativeModel, Part
                # Initialize Vertex AI
                PROJECT_ID = "" # add proj id
                vertexai.init(project=PROJECT_ID, location="europe-west8")
                model = GenerativeModel(model_name="gemini-2.5-pro",
                                        system_instruction=[instruction1])  
                response = model.generate_content([instruction],
                                                    generation_config={
                                                    "temperature": 0.0,  # Deterministic
                                                    "top_p":0.01,
                                                    "top_k":1,
                                                    "candidate_count":1,
                                                    "max_output_tokens": 4000}  # Limit response length
                                                )
                content = response.text
                
            elif model_name=='gpt':
                client = model[0]
                version = model[1]
                
                response = client.chat.completions.create(
                model=version,
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": instruction1}]},
                        {"role": "user", "content": [{"type": "text", "text": instruction}]}
                    ],
                temperature=0.0,
                max_tokens=6000
                )
                content = response.choices[0].message.content


            print('CONTENT:', content)

            
            match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            # print('MATCH',match.group(1))
            if match:
                json_str = match.group(1)  # Extract the JSON part
                try:
                    content_clean = json.loads(json_str)  # Convert to Python dictionary
                    print("Parsed JSON successfully:", content_clean)  # Successfully parsed JSON
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    content_clean = {"reasoning_steps": [], "filtered_answers": ["JSONDecodeError Parsing failed"]}
            else:
                print("No JSON found")
                content_clean = {"reasoning_steps": [], "filtered_answers": ["No Json found"]}

            # Extract final values
            reasoning_steps = content_clean.get("reasoning_steps", [])
            filtered_list = content_clean.get("filtered_answers", ["No answer found"])


            if filtered_list:
                filtered_answer_list[key] = filtered_list
                reasoning[key] = reasoning_steps

                print(f"Question: {question}")
                print(f"Attribute values: {answer_list}")
                print(f"Filtered attribute values: {filtered_answer_list[key]}")
                print()
            else:
                print('filtered list was empty!', c)
                print(f"Question:{question}, prompt: {entity_or_prompt}")
                filtered_answer_list[key] = filtered_list
                reasoning[key] = reasoning_steps
                c+=1


        # Create DataFrame from filtered_answer_list
        filtered_df = pd.DataFrame({
                'entity': [entity for (entity, _) in filtered_answer_list.keys()],
                'question': [question for (_, question) in filtered_answer_list.keys()],
                'answer_list': [filtered_answer_list[(entity, question)] for (entity, question) in filtered_answer_list.keys()],
                'reasoning': [reasoning[(entity, question)] for (entity, question) in reasoning.keys()]
        })

        # Get unique rows from df with the additional columns
        df_unique = df[['entity', 'question', 'entity_id', 'question_id', 'prompt', 'prompt_id']].drop_duplicates()
        
        # Merge filtered_df with df_unique on ['entity', 'question']
        final_df = filtered_df.merge(df_unique, on=['entity', 'question'], how='left')

        # final_df = pd.concat([df1, final_df], ignore_index=True)
        final_df.sort_values(['entity', 'question'], inplace=True)
        # Reorder columns
        final_df = final_df[['entity', 'question', 'entity_id', 'question_id', 'prompt', 'prompt_id', 'answer_list', 'reasoning']]

        # Save to CSV without index
        final_df.to_csv(dataset_path+ "/filtered_answers.csv", index=False)


def main():

    parser = argparse.ArgumentParser(
        description="Generate answer lists."
    )

    parser.add_argument(
        "--model_name",
        choices=[
            'gemini', 'qwen', 'gpt', 'gemini2.5', 'mistral'
        ],
        default='qwen',
        help="Name of the model you want to assess."
    )


    args = parser.parse_args()


    # Load the model 
    
    if args.model_name == 'qwen':     
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

    
    elif args.model_name == 'gemini':
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
        # Initialize Vertex AI
        PROJECT_ID = "proj_id"
        
        model = None
        processor = None
        
    elif args.model_name == 'gemini2.5':
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
        # Initialize Vertex AI
        PROJECT_ID = "proj_id"
        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-2.5-pro")
        processor = None
    
    elif args.model_name =='gpt':
        from openai import OpenAI
        from utils import load_oai_key
        client = OpenAI(api_key=load_oai_key())    
        version = "gpt-4o-2024-08-06"
        model = [client, version]
        processor=None
        
    elif args.model_name =='mistral':
        
        model = None
        processor=None

    dataset_path = os.path.join('datasets', args.model_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

        generate_answer_lists(model, processor, dataset_path, args.model_name)
        filter_answer_lists(model, processor, dataset_path, args.model_name)


if __name__ == "__main__":
    main()
