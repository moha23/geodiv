import argparse
import pandas as pd

def generate_attributes(model, processor, save_path, model_name):
    
    df = pd.read_csv("entity_prompts.csv")
    
    entity_list = df['entity'].tolist()
    entity_ids = df['entity_id'].tolist()
    regions = df['region'].to_list()
    # List to store new rows with entities and their questions
    entity_questions = []

    seen_ids = set()

    for c_id, entity, region in zip(entity_ids, entity_list, regions):
            
        if c_id in seen_ids:
            continue
        
        instruction1 = f"""
                    You are a helpful assistant.
                    Help me ask questions about images that depict certain entities.
                    I will provide you a entity. Your task is to analyze the entity's typical visual attributes and generate **clear and simple questions** about the entity. Your questions should involve concrete attributes and be answerable purely by visually inspecting the image.

                    Do NOT ask follow-up or compound questions within the same question.
                    Do NOT ask questions that cannot be answered by visually inspecting the image or require inference or external context beyond what is shown.
                    Do NOT ask more than 6 questions.

                    Here's an example:
                    **entity**: a house
                    **questions**:
                    1. What is the type of the house?
                    2. What primary construction material is used for the house walls?
                    3. What type of roof does the house have?
                    4. Is the house single-storey or multi-storey?
                    5. What kind of ground cover is visible in front of or around the house?
                    
                    """

        instruction = f"""
                    entity: {entity}
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
            output_ids = model.generate(**inputs, max_new_tokens=2000, temperature=0.01)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            content = output_text[0]

                
        elif model_name=='mistral':
            import torch
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            
            messages = [
                {"role": "system", "content": instruction1},
                {"role": "user", "content": [{"type": "text", "text": instruction}]},
            ]
            
            tokenized = processor.encode_chat_completion(ChatCompletionRequest(messages=messages))
            input_ids = torch.tensor([tokenized.tokens])
            attention_mask = torch.ones_like(input_ids)

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2000,
            )[0]

            content = processor.decode(output[len(tokenized.tokens) :])
            
        elif model_name=='llama':
            
                conversation = [{
                        "role":"user",
                        "content":[{"type":"text","text":instruction1+'\n'+instruction}]
                        }]

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2000
                )
                
                content = processor.decode(outputs[0])    

        elif model_name=='gemini':
            
                import vertexai
                from vertexai.generative_models import GenerativeModel, Part
                # Initialize Vertex AI
                PROJECT_ID = "proj_id" # replace
                vertexai.init(project=PROJECT_ID, location="europe-west8")
                model = GenerativeModel(model_name="gemini-2.5-pro",
                                        system_instruction=[instruction1])
                try:
                    response = model.generate_content([instruction], generation_config={
                                                        "temperature": 0.0,  # Deterministic
                                                        "max_output_tokens": 2000}  # Limit response length
                                                        )
                    content = response.text   
                except Exception as e:
                    import time
                    time.sleep(10)
                    response = model.generate_content([instruction], generation_config={
                                                        "temperature": 0.0,  # Deterministic
                                                        "max_output_tokens": 2000}  # Limit response length
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
                temperature=0.0
            )
            content = response.choices[0].message.content


        print(content)
        
        lines = content.split('\n')
        questions = [line[line.index(' ') + 1:] for line in lines if line.startswith(tuple(str(i) + '.' for i in range(1, 10)))]
        
        for question in questions:
            entity_questions.append({
                'entity_id': c_id,
                'entity': entity,
                'question': question
            })
            
        seen_ids.add(c_id)
        
    entity_questions_df = pd.DataFrame(entity_questions)
    # add question_id column
    entity_questions_df['question_id'] = entity_questions_df.index
    entity_questions_df.to_csv(save_path + f"/entity_questions_org_{model_name}.csv", index=False)

def main():

    parser = argparse.ArgumentParser(
        description="Generate Questions"
    )

    parser.add_argument(
        "--model_name",
        choices=[
            'gemini', 'qwen', 'gpt', 'mistral', 'llama'
        ],
        default='gemini',
        help="Name of the model you want to assess."
    )

    args = parser.parse_args()


    # Load the model 
    
    if args.model_name == 'qwen':     
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
        print('visible devices', torch.cuda.device_count())
        model_path = "../models--Qwen--Qwen2.5-VL-32B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")


    elif args.model_name == 'gemini':
        model = None
        processor = None

    
    elif args.model_name =='gpt':
        from openai import OpenAI
        from utils import load_oai_key
        client = OpenAI(api_key=load_oai_key())    
        version = "gpt-4o-2024-08-06"
        model = [client, version]
        processor=None
        
    elif args.model_name =='mistral':
        import torch
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from huggingface_hub import hf_hub_download
        from transformers import Mistral3ForConditionalGeneration
        
        model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        tokenizer = MistralTokenizer.from_hf_hub(model_id)

        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        
        model = model
        processor=tokenizer
        
    elif args.model_name == 'llama':
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        import torch
        from huggingface_hub import login
        
        login()

        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = model.to(dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_id)


    save_path = './'

    generate_attributes(model, processor, save_path, args.model_name) # questions for each entity

if __name__ == "__main__":
    main()
