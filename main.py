company_description = """TechSolutions Inc. is a mid-sized technology company specializing in software solutions for the healthcare sector. The company was founded in 2015 and has experienced significant growth over the years, currently generating $50 million in annual revenue. They have a diverse range of products and services, including electronic health record (EHR) systems, telemedicine platforms, and healthcare data analytics tools.
In the past fiscal year, TechSolutions Inc. reported a 15% increase in revenue, amounting to $7.5 million, and a 12% increase in net profit, totaling $4.5 million. The company's current market share is at 20%, with three major competitors holding a combined market share of 70%. Their most recent product launch, the TeleHealth platform, resulted in a 10% increase in sales, bringing in an additional $5 million in revenue, and received 80% positive customer feedback.
TechSolutions Inc. has a workforce of 200 employees, with 60 employees working in research and development, 80 employees in sales and marketing, and the remaining 60 employees in administrative and support roles. The company provides regular training and development opportunities for its employees, with a focus on leadership and technical skills. The average tenure of employees at TechSolutions Inc. is 3.5 years, and the company has a 90% retention rate.
The company is considering several new projects for the next fiscal year, including an expansion into the European market, which is projected to generate an additional $10 million in annual revenue. They are also exploring the development of a new EHR system for small healthcare practices, aiming to capture 10% of this niche market, and a healthcare data analytics platform targeting insurance companies, which is forecasted to add another $8 million in revenue within two years. TechSolutions Inc. also plans to raise $15 million in additional funding to support these projects, either through equity investments or business loans.
The company's current cash reserves are at $10 million, with a long-term debt of $5 million at an average interest rate of 4%. TechSolutions Inc.'s gross profit margin is 60%, and its operating margin is 30%. The company has an annual R&D budget of $3 million, a marketing budget of $4 million, and allocates 10% of its revenue to employee training and development."""

questions_json = """{
"rating_or_scoring_questions": [
{
"question": "Considering TechSolutions Inc.'s financial performance, market presence, and employee development initiatives, rate the company's overall attractiveness as an investment opportunity on a scale of 1 to 10."
},
{
"question": "On a scale of 1 to 10, how would you rate TechSolutions Inc.'s overall performance in the past fiscal year, considering their revenue growth, net profit increase, and market share?"
},
{
"question": "Considering the 10% increase in sales and 80% positive customer feedback, rate the success of the TeleHealth platform launch on a scale of 1 to 5."
},
{
"question": "Based on the company's employee training and development opportunities, rate the effectiveness of its talent development strategy on a scale of 1 to 10."
},
{
"question": "On a scale of 1 to 5, how well do you think TechSolutions Inc. is positioned to compete with the three major competitors in the healthcare software market?"
}
],
"categorical_questions": [
{
"question": "Based on TechSolutions Inc.'s current financial situation and growth prospects, which of the following strategies should the company prioritize? Please choose one option: (1) Focus on organic growth through reinvesting profits (2) Pursue inorganic growth through mergers and acquisitions (3) Combine organic growth with selective mergers and acquisitions (4) Diversify into new markets while maintaining current growth strategies"
},
{
"question": "Considering TechSolutions Inc.'s product portfolio and market presence, should the company primarily focus on (1) enhancing its existing products and services, or on (2) developing new and innovative solutions for the healthcare sector?"
},
{
"question": "Given the company's current financial situation, should TechSolutions Inc. seek additional funding through (1) equity investments, (2) business loans, or (3) a combination of both?"
},
{
"question": "Based on the company's growth strategy, should TechSolutions Inc. prioritize (1) expansion into the European market, (2) developing the new EHR system, or (3) creating the healthcare data analytics platform?"
},
{
"question": "Considering the company's current employee structure, should TechSolutions Inc. (1) hire more employees in research and development, (2) sales and marketing, or (3) administrative and support roles?"
}
],
"estimation_questions": [
{
"question": "Assuming TechSolutions Inc. successfully expands into the European market, estimate the percentage of total company revenue that will be generated from this new market within the first two years."
},
{
"question": "Given the TechSolutions Inc. 15% increase in revenue and 12% increase in net profit, estimate TechSolutions Inc.'s potential net profit growth in the next fiscal year (in percentage)."
},
{
"question": "Considering the company's current market share of 20% and the competitors' combined market share of 70%, estimate TechSolutions Inc.'s potential market share in the next three years (in percentage)."
},
{
"question": "Given the TechSolutions Inc. plans to raise additional funding, estimate the percentage of funds that should be allocated to the following project: Healthcare data analytics platform targeting insurance companies."
},
{
"question": "Based on the company's current workforce distribution (30% in R&D, 40% in sales and marketing, and 30% in administrative and support roles), estimate the percentage increase in the workforce required for sales and marketing department to support the company's growth over the next three years."
}
],
"decision_thresholds": [
{
"question": "TechSolutions Inc. is planning to expand its product line. What is the minimum projected annual revenue increase (in percentage) that would justify the development and launch of a new product in the healthcare software market?"
},
{
"question": "TechSolutions Inc. is considering raising additional funding through business loans. At what annual interest rate (in percentage) would you recommend not pursuing this funding option?"
},
{
"question": "TechSolutions Inc. is evaluating the potential return on investment (ROI) for the new EHR system project. What is the minimum ROI (in percentage) the company should expect to justify pursuing the project?"
},
{
"question": "Assuming TechSolutions Inc. needs to raise $5 million to fund its expansion plans, what is the maximum percentage of company ownership that it should be willing to give up in exchange for the required funding, considering its current valuation of $25 million?"
},
{
"question": "TechSolutions Inc. currently has a customer satisfaction rate of 85% for its products. In order to maintain its market position and strengthen customer loyalty, what should be the minimum percentage of positive customer feedback the company should aim for in the upcoming year?"
}
]
}"""

import numpy as np
import openai
import json
from tqdm import tqdm
import time

# Set up OpenAI API credentials
openai.api_key = "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"
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

import signal

class TimeoutException(Exception):
    pass
    
def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")
    
def call_with_timeout(func, args=(), kwargs={}, timeout=2):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        result = None
    finally:
        signal.alarm(0)
    return result


def chatgpt_complete(messages, model, temperature):
    response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=2
    )
    
    return response["choices"][0]["message"]["content"]

def chatgpt_complete_timeout(messages, model, temperature):
    return call_with_timeout(chatgpt_complete, args=(messages, model, temperature))

def get_answer_openai(question, model, temperature):
    
    messages=[
        {"role": SYSTEM, "content": company_description},
        {"role": SYSTEM, "content": "Your task is to give the best estimate of the answer the provided questions based on the given context to the best of your ability. You must only return an INTEGER number."},
        {"role": USER, "content": question+"\n\n I NEED YOU TO PROVIDE A CONCISE ANSWER TO THE ABOVE QUESTION. RETURN ONLY MACHINE-READABLE ANSWER, NO EXPLANATION REQUIRED. ONLY A NUMER."}
    ]
    result = None
    try:
        result = chatgpt_complete_timeout(messages, model, temperature)
        if result is None:
            #Try one more time
            result = chatgpt_complete_timeout(messages, model, temperature)
    except:
        result = chatgpt_complete_timeout(messages, model, temperature)
    print(result)
    return result

def get_answer(question, model, temperature):
    messages=[
        {"role": SYSTEM, "content": company_description},
        {"role": SYSTEM, "content": "Your task is to give the best estimate of the answer the provided questions based on the given context to the best of your ability. You must only return an INTEGER number."},
        {"role": USER, "content": question+"\n\n I NEED YOU TO PROVIDE A CONCISE ANSWER TO THE ABOVE QUESTION. RETURN ONLY MACHINE-READABLE ANSWER, NO EXPLANATION REQUIRED. ONLY A NUMER."}
    ]
    result = None
    try:
        result = chatgpt_complete_timeout(messages, model, temperature)
        if result is None:
            #Try one more time
            result = chatgpt_complete_timeout(messages, model, temperature)
    except:
        result = chatgpt_complete_timeout(messages, model, temperature)
    print(result)
    return result


def parse_answer(answer):
    try:
        if answer.isnumeric():
            return int(answer)
        
        answer = answer.strip(".")
        if answer.isnumeric():
            return int(answer)
        
        answer = answer.lower()
        if "answer:" in answer:
            answer = answer.split("answer:")[1].strip()
            
            if "," in answer:
                    return [int(x) for x in answer.split(",")]
            elif "." in answer:
                    return [int(x.split(".")[0]) for x in answer.split()]
            elif "-" in answer:
                    return [int(x.strip()) for x in answer.split("-")]
            
            return int(answer)
        
        if "%" in answer:
            return int(answer.split("%")[0].strip())
        
        if "," in answer:
            return [int(x) for x in answer.split(",")]
        
        return None
    except: 
            return None
        
def load_responses(filepath):
    try:
        with open(filepath, 'r') as infile:
            return json.load(infile)
    except FileNotFoundError:
        return []

def save_responses(responses, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(responses, outfile, indent=4)

def check_section_exists(results, temperature, model, iteration):
    for section in results:
        if (section['temperature'] == temperature and section['model'] == model and section['iteration'] == iteration):
            return True
    return False

def check_answer_exists(json_file, question, temperature, model, iteration):
    data = []
    try:
        with open(json_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        pass
        
    for d in data:
        if d['temperature'] == temperature and d['model'] == model and d['iteration'] == iteration:
            for q in d['responses']:
                for response in d['responses'][q]:
                    if response['question'] == question:
                        if 'answer' in response:
                            return response['answer']
    return None

def answer_questions(questions_json, model="gpt-3.5-turbo", temperature=0.8, iteration=0):
    data = json.loads(questions_json)
    saved_responses = load_responses('responses.json')

    responses = {key: [] for key in data}
    for question_type, questions in data.items():
        for question_obj in questions:
            time.sleep(0.1)
            question = question_obj["question"]

            # Check if the answer is already in saved_responses
            existing_answer = check_answer_exists('responses.json', question, temperature, model, iteration)
            if existing_answer is not None:
#               print("Answer exists for model {}, temp {}, iter {}".format(model, temperature))
                answer = existing_answer
            else:
                initial = get_answer_openai(question, model, temperature)
                answer = parse_answer(initial)
                if answer is None:
                    print("COULD NOT PARSE THE ANSWER")
                    print(question)
                    print(initial)
                print(temperature)
                print(iteration)
                print({"question": question, "answer": answer})
            
            responses[question_type].append({"question": question, "answer": answer})
    return responses

def collect_answers(model):
    results = load_responses('responses.json')
    
    for temperature in np.arange(0.0, 1.0, 0.1):
        temperature = round(temperature, 1)
        print(model)
        print("Temp = " + str(temperature))
        for i in tqdm(range(2)):
            all_responses = {}
            all_responses["temperature"] = temperature
            all_responses["model"] = model
            all_responses["iteration"] = i
            
            responses = answer_questions(questions_json, model, temperature, i)
            all_responses["responses"] = responses
            results.append(all_responses)
            
            save_responses(results, 'responses.json')
            
if __name__ == "__main__":
    collect_answers("gpt-4")
    